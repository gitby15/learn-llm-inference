import uvicorn
def launch_server():
    host = '127.0.0.1'
    port = 8888
    uvicorn.run(
        "learn_llm_inference.http_server.router:app",
        host=host,
        port=port,
        # reload=True,
        # reload_dirs=["src"],
    )


def main():
    launch_server()
    

if __name__ == "__main__":
    main()
