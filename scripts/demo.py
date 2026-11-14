"""Click-to-go demo: a web page where a click on the map sends the worm there.

Open the printed address in a browser. The server listens on localhost only.
The simulation runs on the main thread (MuJoCo rendering needs it). HTTP runs in a daemon thread.

`--smoke` starts the server, reads `/api/state` once, prints it as JSON and exits 0.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import webbrowser

from worm.demo import DemoSession, MapRequest, load_controller
from worm.server import SimLoop, make_server, serve_in_thread


def build_session(policy: str | None, render: bool) -> DemoSession:
    """Build a session with a first map. Fall back to no rendering if the renderer fails."""
    controller = load_controller(policy)
    try:
        session = DemoSession(controller, render=render)
        session.new_map(MapRequest())
    except Exception as error:
        if not render:
            raise
        print(f"rendering is unavailable ({error}); continuing without the 3D view")
        session = DemoSession(controller, render=False)
        session.new_map(MapRequest())
    return session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None, help='a policy folder or "scripted"')
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True,
                        help="serve a 3D follow-camera frame")  # fmt: skip
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    parser.add_argument("--smoke", action="store_true", help="serve /api/state once and exit")
    args = parser.parse_args()

    session = build_session(args.policy, args.render and not args.smoke)
    loop = SimLoop(session)
    httpd = make_server(loop, args.host, args.port)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    serve_in_thread(httpd)

    if args.smoke:
        loop.start()
        state: dict[str, object] = {}
        for _ in range(100):  # the loop publishes its first state a moment after it starts
            with urllib.request.urlopen(url + "api/state", timeout=30) as reply:
                state = json.loads(reply.read())
            if "policy" in state:
                break
            time.sleep(0.1)
        print(json.dumps(state))
        httpd.shutdown()
        loop.stop()
        loop.join(timeout=10)
        raise SystemExit(0)

    print(f"policy: {load_controller(args.policy).name}\nopen {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        loop.run()  # on the main thread, so rendering works on macOS
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        loop.stop()
