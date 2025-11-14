import logging
import sys
from fastapi import FastAPI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/test")
def test_endpoint():
    logger.info("=== LOGGER INSIDE ENDPOINT ===")
    print("=== PRINT INSIDE ENDPOINT ===", flush=True)
    sys.stdout.write("=== STDOUT WRITE INSIDE ENDPOINT ===\n")
    sys.stdout.flush()
    
    # Принудительная запись в файл
    with open("debug_test.txt", "a") as f:
        f.write("=== FILE WRITE INSIDE ENDPOINT ===\n")
    
    return {"message": "test"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None, access_log=False)