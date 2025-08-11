# Development

Launch ComfyUI in dev mode (restart on file change):
`watchmedo auto-restart -d . -p '*.py' -R  -- python main.py --listen --port 8080 --preview-method auto`

(Intall watchmedo - if not already installed: `pip install watchdog`)
