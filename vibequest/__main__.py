import uvicorn
from .server import create_app

uvicorn.run(create_app(), host="0.0.0.0", port=8080, log_level="info")
