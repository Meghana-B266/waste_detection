"""
Entry point for the WEB version of the system.

This replaces the old local OpenCV window (run_complete_system.py) with a
proper web server: any browser on the same network — including phones and
tablets — can view the live stream and dashboard by visiting this
computer's address, with nothing installed on the viewing device.

Run with:
    python run_web_server.py

Then open:
    http://localhost:8000                (on this computer)
    http://<this-computer's-LAN-IP>:8000  (from any other device on the same Wi-Fi)

To find this computer's LAN IP:
    Windows:      ipconfig            → look for "IPv4 Address"
    Mac/Linux:    ifconfig | grep inet

Note: this is for use on a trusted local network (home/office Wi-Fi). It is
not hardened for exposure directly to the public internet — see the
"Deploying beyond your local network" section of the README for that.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
