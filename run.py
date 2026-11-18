from app import create_app
app = create_app()
if __name__ == "__main__":
    print("Starting Potato Harvest Flask App...")
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True,
        use_reloader=False
    )
