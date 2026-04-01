
import asyncio
from app.helpers.loop_helpers import set_uvloop
SERVER_LOOP = set_uvloop()

from app.config.config import *
from app.config.config import from_env

import app.helpers.polars_hyper_plugin
import polars as pl

import tracemalloc
import time, uuid, os
from typing import Optional, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin
from datetime import timedelta

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from jinja2.exceptions import TemplateNotFound
from app.helpers.async_jinja import AsyncJinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.corswebsocket import WSSafeCORSMiddleware
from fastapi.staticfiles import StaticFiles

from starlette.responses import Response

from app.helpers.process_helpers import set_process_affinity_high
from app.helpers.asyncThreadExecutor import AsyncThreadExecutor
from app.logs.logging import log

# SERVER GLOBAL --------------------------------------------------------
ENV = from_env('ENV', "UAT")
CACHE_PORT = from_env('CACHE_PORT', 8001)
VITE_HOST = from_env('VITE_HOST', 'localhost')
VITE_PORT = from_env('VITE_PORT', 7778)
DB_PATH = from_env('DB_PATH', './app/data/portfolioToolDev.db')
APP_VERSION = from_env('APP_VERSION', 0.0)
OS_SYSTEM = from_env('OS_SYSTEM', 'Windows').title()
# -----------------------------------------------------------------------


if OS_SYSTEM == 'Windows':
    set_process_affinity_high()

def get_interval(override=None, default=-1):
    return override or from_env('SYNC_INTERVAL', default=default, dtype=int)

# -----------------------------------------------------------------------
# Shared Globals
# -----------------------------------------------------------------------

KDBCACHE_PROCESS = None
PB, PBA, PBF = None, None, None
MM, KDB, S3, DB, PM, KSM = None, None, None, None, None, None
DB_LOCAL, DB_SHARED = None, None
S3CACHE, KDBCACHE = None, None
MONITOR, TRACE, TASKCONTEXT, THREADS, = None, None, None, None

def get_resource_monitor():
    global MONITOR
    from app.services.server.resourceMonitor import ResourceMonitor
    if ('MONITOR' in globals()) and (not MONITOR is None): return MONITOR
    MONITOR = ResourceMonitor(
        check_interval = 20,
        detailed_interval = 120,
        history = 40,
        freeze_threshold = 15
    )

    return MONITOR


# -----------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------

from app.helpers.tracker import track
def startup(name): return track(name, log.startup)
def shutdown(name): return track(name, log.shutdown)


@startup('Resource Monitor')
async def init_monitor(ctx):
    global MONITOR
    if from_env('MONITOR_RESOURCES', dtype=bool, default=True):
        MONITOR = get_resource_monitor()
        ctx.spawn(MONITOR.monitor_resources())
        return MONITOR
    return NotImplementedError, None

@shutdown('Resource Monitor')
async def shutdown_monitor():
    global MONITOR
    if MONITOR is not None:
        await MONITOR.shutdown()
        del MONITOR
    return True


@startup("Async Thread Executor")
async def init_thread_executor():
    global THREADS
    THREADS = AsyncThreadExecutor(name="server-executor")
    THREADS.start()
    return THREADS

@shutdown('Async Thread Executor')
async def shutdown_thread_executor():
    global THREADS
    if ("THREADS" in globals()) and (THREADS is not None):
        THREADS.shutdown()
        del THREADS
    return True

def get_threads():
    global THREADS
    if ("THREADS" not in globals()) or (THREADS is None):
        THREADS = AsyncThreadExecutor(name="server-executor")
        THREADS.start()
    return THREADS

@startup("Database")
async def init_db():
    global DB
    DB = get_db()
    await DB.connect()
    return DB

@shutdown('Database')
async def shutdown_db():
    global DB
    if DB is not None:
        await DB.disconnect()
        del DB
    return True

@startup("Phonebook")
async def init_phonebook():
    global PB, PBA, PBF
    try:
        PB = await get_pb()
        PBA = get_pba()
        PBF = await get_pbf()
    except Exception as e:
        PB, PBA, PBF = None, None, None
        return e, (PB, PBA, PBF)
    return PB, PBA, PBF

@shutdown('Phonebook')
async def shutdown_phonebook():
    global PB, PBA, PBF
    if PB is not None:
        del PB; del PBA; del PBF
    return True

@startup("KDB Worker")
async def init_kdb():
    global KDB
    KDB = get_kdb()
    KDB.start()
    return KDB

@shutdown('KDB Worker')
async def shutdown_kdb():
    global KDB
    if KDB is not None:
        KDB.shutdown_all()
        del KDB
    return True

@startup("S3 Cache Service")
async def init_s3_cache():
    global S3CACHE
    S3CACHE = get_s3_cache()
    return S3CACHE

@shutdown('S3 Cache Service')
async def shutdown_s3_cache():
    global S3CACHE
    if S3CACHE is not None:
        await S3CACHE.save_cache_to_disk()
        del S3CACHE
    return True

@startup("KDB Quoteevent Cache")
async def init_kdb_qe_cache():
    global KDBCACHE_PROCESS, KDBCACHE
    if not from_env('KDB_CACHE_PROCESS', dtype=bool, default=False):
        return NotImplementedError, (None, None)
    interval = get_interval()
    if interval > 0:
        import multiprocessing
        from app.services.cache.kdb_cache_service import run_cache_service
        KDBCACHE_PROCESS = multiprocessing.Process(target=run_cache_service, daemon=True)
        KDBCACHE_PROCESS.start()
        time.sleep(1)
        KDBCACHE = get_cache_client()
        return KDBCACHE, KDBCACHE_PROCESS
    return NotImplementedError, (None, None)

@shutdown('KDB Quoteevent Cache')
async def shutdown_kdb_qe_cache():
    global KDBCACHE_PROCESS, KDBCACHE
    if KDBCACHE is not None:
        await KDBCACHE.close()
        del KDBCACHE

    if (KDBCACHE_PROCESS is not None) and KDBCACHE_PROCESS.is_alive():
        KDBCACHE_PROCESS.terminate()  # Sends SIGTERM
        try:
            KDBCACHE_PROCESS.join(timeout=5)  # Wait for it to exit
        except Exception as e:
            return e, False
        finally:
            del KDBCACHE_PROCESS
    return True

@startup("KSM Sync Service")
async def init_ksm():
    global KSM
    KSM = get_ksm()
    from app.services.loaders.kdb_queries_dev_v2 import book_maps
    maps = await book_maps(regions=("US", "EU", "SGP")) # prime cache
    return KSM

@startup("KSM Sync Service")
async def shutdown_ksm():
    global KSM
    if KSM is not None:
        KSM.stop()
        del KSM
    return True

@startup("Connection Manager")
async def init_mm():
    global MM
    MM = get_mm()
    await MM.init()
    return MM

@shutdown("Connection Manager")
async def shutdown_mm():
    global MM
    if MM is not None:
        await MM.shutdown()
        del MM
    return True

@startup("Tracemalloc")
async def init_tracemalloc():
    global TRACE
    if from_env('TRACE_RESOURCES', False):
        tracemalloc.start()
        TRACE = True
        return TRACE
    return NotImplementedError, False

@shutdown("Tracemalloc")
async def shutdown_tracemalloc():
    global TRACE
    if TRACE is not None:
        tracemalloc.stop()
    return True

@startup("TaskContext")
async def init_ctx():
    global TASKCONTEXT
    TASKCONTEXT = get_ctx()
    return TASKCONTEXT

@shutdown("TaskContext")
async def shutdown_ctx():
    global TASKCONTEXT
    from app.helpers.taskContext import TaskContext
    if TASKCONTEXT is not None:
        await TASKCONTEXT.close()
        del TASKCONTEXT
    await TaskContext.shutdown()
    return True

@asynccontextmanager
async def lifespan(my_app: FastAPI):

    await log.startup('Entering lifespan')
    global KDB, MM, S3, DB, PM, KSM, S3CACHE
    global KDBCACHE, KDBCACHE_PROCESS
    global PB, PBA, PBF
    global MONITOR, TRACE, TASKCONTEXT
    global SERVER_LOOP

    # Set UVLOOP on this process
    # from app.helpers.loop_helpers import set_uvloop
    # SERVER_LOOP = set_uvloop(SERVER_LOOP)

    # Garbage tweaks
    if from_env('TWEAK_GC', dtype=bool, default=1):
        import gc
        gc.collect(2)
        gc.freeze()
        allocs, gen1, gen2 = gc.get_threshold()
        allocs = 50_000
        gen1 *= 2
        gen2 *= 2
        gc.set_threshold(allocs, gen1, gen2)

    try:
        TASKCONTEXT = await init_ctx()
        MONITOR = await init_monitor(TASKCONTEXT)
        THREADS = await init_thread_executor()
        DB = await init_db()
        PB, PBA, PBF = await init_phonebook()
        KDB = await init_kdb()
        S3CACHE = await init_s3_cache()
        KDBCACHE_PROCESS, KDBCACHE = await init_kdb_qe_cache()
        KSM = await init_ksm()
        MM = await init_mm()
        TRACE = await init_tracemalloc()

        from app.services.portfolio.ibBot import ensure_started as _ibbot_start
        try:
            await _ibbot_start()
            await log.startup("IBBot started")
        except Exception as e:
            await log.error(f"IBBot failed to start: {e}")

        # Final book up
        KSM.start()
        await log.startup(f"Startup is complete: {APP_URL}")

        yield  # Application runs here
    finally:

        # IBBot shutdown
        from app.services.portfolio.ibBot import ensure_stopped as _ibbot_stop
        try:
            await _ibbot_stop()
            await log.shutdown("IBBot stopped")
        except Exception as e:
            await log.error(f"IBBot shutdown error: {e}")

        await shutdown_tracemalloc()
        await shutdown_db()
        await shutdown_s3_cache()
        await shutdown_mm()
        await shutdown_kdb_qe_cache()
        await shutdown_ksm()
        await shutdown_kdb()
        await shutdown_thread_executor()
        await shutdown_monitor()
        await shutdown_ctx()

        await log.success("Shutdown complete")

        # Shutdown loggers
        log.shutdown_all()

from app.config.router_settings import SWAGGER_UI_PARAMETERS

app = FastAPI(
    title = "Barclays Portfolio Webtool",
    summary = "REST API for the portfolio pricing tool.",
    version = str(round(APP_VERSION,5)),
    contact = {
        "name": "Cade Zawacki",
        "email": "Cade.Zawacki@barclays.com"
    },
    lifespan=lifespan,
    swagger_ui_parameters=SWAGGER_UI_PARAMETERS,
)

def check_event_loop_status():
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running(): return 1
        else: return 0
    except RuntimeError:
        return -1

def get_cache_client(port=None):
    global KDBCACHE
    from app.services.kdb.tickerplantClient import KdbCacheClient
    port = port if port is not None else CACHE_PORT
    url = f"http://127.0.0.1:{port}"

    if not 'KDBCACHE' in globals():
        KDBCACHE = KdbCacheClient(base_url=url)
    if KDBCACHE is None:
        KDBCACHE = KdbCacheClient(base_url=url)
    return KDBCACHE

kdb_lock = False
def get_kdb():
    global KDB, kdb_lock
    from app.services.server.serverManager import KDBServer
    if (not 'KDB' in globals()) or (KDB is None):
        kdb_lock = True
        KDB = KDBServer(auto_start=False)
        KDB.start()
    return KDB

def get_ctx():
    global TASKCONTEXT
    if (not 'TASKCONTEXT' in globals()) or (TASKCONTEXT is None):
        from app.helpers.taskContext import TaskContext
        TASKCONTEXT = TaskContext()
    return TASKCONTEXT

def get_s3_cache():
    from app.services.portfolio.s3 import S3Cache
    global S3CACHE
    if ('S3CACHE' not in globals()) or (S3CACHE is None):
        S3CACHE = S3Cache()
    return S3CACHE


def get_mm():
    from app.services.redux.connectionManager import ConnectionManager
    global MM
    if (not 'MM' in globals()) or (MM is None):
        db = get_db()
        MM = ConnectionManager()
    return MM

def get_ksm():
    global KSM, DB, MM, KDB
    if ('KSM' not in globals()) or (KSM is None):
        from app.services.server.serverManager import SyncServer
        DB = get_db(); KDB = get_kdb(); MM = get_mm()
        KSM = SyncServer(maintenance_interval=get_interval())
        KSM.start()
    return KSM

def get_db():
    global DB
    if ('DB' not in globals()) or (DB is None):
        from app.services.storage.portfolioManager import PortfolioManager
        DB = PortfolioManager(DB_PATH)
    return DB

def get_s3(force=False):
    from app.services.portfolio.s3 import S3Service
    global S3

    if force and ('S3' in globals()) and (S3 is not None):
        executor = get_threads()
        try:
            fut = executor.submit(S3.close)
            fut.result()
        except Exception as e:
            log.error('Failed to stop S3 instance')
        finally:
            S3 = None
    if ('S3' not in globals()) or (S3 is None):
        S3 = S3Service()

    return S3

async def get_pb(trading_only=False, *, lazy=True, raw=False, force=False):
    from app.services.portfolio.users import request_phonebook
    pb = await request_phonebook(trading_only=trading_only, force=force)
    if raw: return pb
    inners = list(pb.values())
    return pl.LazyFrame(inners) if lazy else pl.DataFrame(inners)

def get_pba():
    from app.services.portfolio.users import search_people
    return search_people

async def get_pbf():
    global PBF
    if ('PBF' not in globals()) or (PBF is None):
        pba = get_pba()
        from app.services.rules.desigMatchV2 import desigNameFuzzyMatchRule
        PBF = desigNameFuzzyMatchRule(pba)
    return PBF

def get_static_folder(base=None, suppress_base=False):
    if suppress_base: return from_env('STATIC_FOLDER', 'static')
    base = base or (Path(__file__).parent.parent if '__file__' in globals() else '/')
    return str(os.path.join(base, from_env('STATIC_FOLDER', 'static')))

def get_templates_folder(base=None, name='app', suppress_base=False):
    if suppress_base: return from_env('TEMPLATE_FOLDER', 'templates')
    base = base or (Path(__file__).parent.parent if '__file__' in globals() else '/')
    return str(os.path.join(base, name, from_env('TEMPLATE_FOLDER', 'templates')))

def get_page_path(name):
    name = name if name.endswith('.html') else name + '.html'
    return str(os.path.join(TEMPLATE_DIR, name))

DEV_MODE = from_env('ENV', 'UAT') in ['UAT', 'DEV', 'QA']
STATIC_DIR = get_static_folder()
TEMPLATE_DIR = get_templates_folder()

templates = AsyncJinja2Templates('app', get_templates_folder(suppress_base=True))

# Middleware ###############################
app.add_middleware(WSSafeCORSMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=1)

def install_html_etag_middleware(
        app: FastAPI,
        *,
        aggressive: bool = False,
        weak: bool = True,
        excluded_prefixes: Optional[Iterable[str]] = None,
):
    excluded = tuple(excluded_prefixes or ("/docs", "/openapi", "/redoc", "/assets", '/ws', '/js'))

    def build_etag_value(request: Request) -> Optional[str]:
        if aggressive:
            raw = uuid.uuid4().hex
        else:
            app_version = getattr(request.app, "version", None)
            raw = f"{app_version}:{request.url.path}" if app_version else request.url.path
        if not raw:
            return None
        tag = f'"{raw}"'
        return f'W/{tag}' if weak else tag

    @app.middleware("http")
    async def html_etag_middleware(request: Request, call_next):
        try:
            if request.method not in ("GET", "HEAD"):
                return await call_next(request)

            path = request.url.path
            if any(path.startswith(pfx) for pfx in excluded):
                return await call_next(request)

            response = await call_next(request)
            if not hasattr(response, "headers"): return response
            if "etag" in response.headers or response.status_code in (204, 304, 101):
                return response

            content_type = (response.headers.get("content-type") or "").lower()
            if "text/html" not in content_type:
                return response

            etag_value = build_etag_value(request)
            if not etag_value: return response

            client_etags = request.headers.get("if-none-match")
            if client_etags:
                tokens = [t.strip() for t in client_etags.split(",") if t.strip()]
                if etag_value in tokens:
                    headers = dict(response.headers)
                    headers["ETag"] = etag_value
                    headers.pop("content-length", None)
                    return Response(status_code=304, headers=headers)

            response.headers["ETag"] = etag_value
            return response
        except Exception as e:
            await log.error(f"Failed to etag response: {e}")

install_html_etag_middleware(app, aggressive=False, weak=True)

@app.middleware("http")
async def cache_control_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        if not hasattr(request, "url"):
            return response

        if path.endswith("/assets/js/sw.js") or path.endswith("/assets/js/filterWorker.js"):
            response.headers["Service-Worker-Allowed"] = "/"
            response.headers["Content-Type"] = "text/javascript"

        elif "/assets/js/" in path:
            response.headers["Service-Worker-Allowed"] = "/"
            response.headers["Content-Type"] = "text/javascript"

        elif "/assets/lottie/" in path:
            response.headers["Cache-Control"] = "public, max-age=43200"
            response.headers["Content-Type"] = "text/json"

        elif "/assets/audio/" in path:
            response.headers["Cache-Control"] = "public, max-age=43200"
            response.headers["Content-Type"] = "audio/mpeg"

        elif ("/assets/img/" in path) or ("/assets/ico/" in path):
            response.headers["Cache-Control"] = "public, max-age=43200"

        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    except Exception as e:
        await log.error(f"Failed to cache response: {e}")
        try:
            await log.error(path.__dict__)
        except Exception:
            pass



##############################################################

# If we are running in PROD, then reference the bundled files directly.
mount_path = '/' + get_static_folder(suppress_base=True)
log.app(message=f"Mounting static endpoint as .{mount_path}", color="gray")
app.mount(mount_path, StaticFiles(directory=get_static_folder()), name="static")

SEVER_INFO = {
    "dev_mode": int(DEV_MODE),
    "static_path": mount_path,
    "vite_host": f"http://{VITE_HOST}:{VITE_PORT}" if DEV_MODE else "",
    "app_version": APP_VERSION
}

# Assets + Cache
app.mount("/assets", StaticFiles(directory="./assets"), name="assets")


@app.get("/", include_in_schema=False)
async def serve_root(request: Request):
    return await templates.TemplateResponse(
        'frame.html',
        {
            "request": request,
            "page": "homepage",
            **SEVER_INFO
        }
    )
def create_route_handler(path_name):
    async def base_handler(request: Request, additional_context:str = ""):
        return await templates.TemplateResponse(
            'frame.html',
            {
                "request": request,
                "page": urljoin(f"{path_name.lower()}/", additional_context).strip('/'),
                **SEVER_INFO
            }, status_code=200
        )
    return base_handler

sidebar_paths = ["homepage", "charts", "upload", "blotter", "logs", "status", "messaging", "hotkeys", "settings", "pt", "404"]
for path in sidebar_paths:
    app.get("/%s/{additional_context:path}" % path, response_class=HTMLResponse, include_in_schema=False)(create_route_handler(path))

app.get("/pt/pt/{additional_context:path}", response_class=HTMLResponse, include_in_schema=False)(create_route_handler("pt"))

@app.get("/templates/{page_name}", include_in_schema=False)
async def serve_page(request: Request, page_name: str = "", context: str = Query("", description="Additional Context to pass to HTML template")):
    page_name = "homepage" if page_name == "" else page_name
    await log.notify(f"Serving: {page_name}")
    try:
        return await templates.TemplateResponse(
            f"{page_name}.html" if not page_name.endswith('html') else page_name,
            {
                "request": request,
                "page": page_name.replace(".html","").split("-")[0],
                "context": context,
                **SEVER_INFO
            },
            status_code=200
        )
    except TemplateNotFound:
        return await error_exception_handler(request, HTTPException(404, "Page not found"))
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    try:
        return await templates.TemplateResponse(
            'frame.html',
            {
                "request": request,
                "page": '404',
                **SEVER_INFO
            }, status_code=200
        )
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))

async def error_exception_handler(request, exc):
    try:
        return await templates.TemplateResponse(
            'error.html',
            {
                "request": request,
                "page": 'error',
                "error_description": "", #str(exc.detail),
                "error_code": 400, #exc.status_code,
                **SEVER_INFO
            },
            status_code=200
        )
    except Exception:
        raise HTTPException(status_code=417, detail="Service Unavailable")

from app.routers import frame
app.include_router(frame.router)

from app.routers import socket
app.include_router(socket.router)

from app.routers import pt
app.include_router(pt.meta_router)
app.include_router(pt.pt_router)

from app.routers import data
app.include_router(data.router)

from app.routers import pricing
if ENV != 'PREP':
    app.include_router(pricing.router)

from app.routers import load
app.include_router(load.router)

from app.routers import messaging
if ENV != 'PREP':
    app.include_router(messaging.router)

from app.routers import users
if ENV != 'PREP':
    app.include_router(users.router)

from app.routers import s3
if ENV != 'PREP':
    app.include_router(s3.router)

from app.routers import analytics
if ENV != 'PREP':
    app.include_router(analytics.router)

from app.routers import health
app.include_router(health.router)

from app.routers import debug
if ENV != 'PREP':
    app.include_router(debug.router)

from app.routers import warnings
if ENV != 'PREP':
    app.include_router(warnings.router)

from app.routers import bored
if ENV != 'PREP':
    app.include_router(bored.router)

from app.routers import dev
if ENV != 'PREP':
    app.include_router(dev.router)

from app.routers import arrow_grid_router
app.include_router(arrow_grid_router.router)

# IBBot management API
from app.routers import ibbot
app.include_router(ibbot.router)

# BVAL Monitoring
from app.services.portfolio.ibBot import create_bval_router
app.include_router(create_bval_router())

# DEPRECIATED -----------------------

# from app.routers import sync
# app.include_router(sync.router)

# from app.routers import refresh
# app.include_router(refresh.router)

# from app.routers import bval
# app.include_router(bval.router)

# -----------------------------------

# if __name__ == "__main__":
#     server = AppServer(app)



