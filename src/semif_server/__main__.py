import uvicorn


def main() -> None:
    uvicorn.run(
        "semif_server.server:app",
        host="127.0.0.1",
        port=8787,
        workers=1,
    )


if __name__ == "__main__":
    main()