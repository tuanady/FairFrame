# from flask import Flask, render_template, request, jsonify
# import requests
#
# app = Flask(__name__)
#
# # --- CONFIGURATION ---
# # Service A: The Detector (DeBERTa)
# DETECTOR_URL = "https://modal-labs-civicmachines--stemagine-classifier-llamaclas-489335.modal.run"
#
# # Service B: The Rewriter (Llama-3)
# REWRITER_URL = "https://modal-labs-civicmachines--fairframe-rewriter-rewriter-rewrite.modal.run"
#
#
# @app.route('/')
# def index():
#     """Render the main page"""
#     return render_template('index.html')
#
#
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """
#     Main Orchestrator:
#     1. Sends prompt to Detector.
#     2. If gaps found -> Sends to Rewriter.
#     3. Returns combined result to UI.
#     """
#     try:
#         data = request.get_json()
#         prompt = data.get('prompt', '').strip()
#
#         if not prompt:
#             return jsonify({'error': 'Prompt cannot be empty'}), 400
#
#         # --- STEP 1: CALL DETECTOR (Service A) ---
#         print(f"📡 Calling Detector: {DETECTOR_URL}")
#         det_response = requests.post(DETECTOR_URL, json={"prompt": prompt})
#
#         if not det_response.ok:
#             raise Exception(f"Detector API Error: {det_response.status_code}")
#
#         det_result = det_response.json()
#         active_gaps = det_result.get("gaps", [])
#
#         # --- STEP 2: CALL REWRITER (Service B) ---
#         # Only call Llama-3 if actual gaps were found
#         if active_gaps:
#             print(f"⚡ Gaps found ({active_gaps}). Calling Rewriter: {REWRITER_URL}")
#
#             payload = {
#                 "original_prompt": prompt,
#                 "gaps": active_gaps
#             }
#
#             rew_response = requests.post(REWRITER_URL, json=payload)
#
#             if rew_response.ok:
#                 rew_data = rew_response.json()
#                 # Merge the rewritten text into the result
#                 det_result["rewritten"] = rew_data.get("rewritten", "Error parsing rewrite.")
#             else:
#                 print(f"❌ Rewriter Error: {rew_response.text}")
#                 det_result["rewritten"] = "Error: Could not generate rewrite."
#         else:
#             print("✅ Safe prompt. Skipping rewrite.")
#             det_result["rewritten"] = None
#
#         return jsonify(det_result)
#
#     except Exception as e:
#         print(f"❌ Server Error: {e}")
#         return jsonify({'error': str(e)}), 500
#
#
# if __name__ == '__main__':
#     print("🚀 Starting FairFrame web server...")
#     print("📍 Open http://localhost:5001 in your browser")
#     app.run(debug=True, host='0.0.0.0', port=5001)

import modal
from flask import Flask, render_template, request, jsonify
import requests
import os

# --- MODAL CONFIGURATION ---
# 1. Define the cloud environment (installs Flask & Requests)
web_image = (
    modal.Image.debian_slim()
    .pip_install("flask", "requests")
    .add_local_dir("templates", remote_path="/root/templates")
    .add_local_dir("static", remote_path="/root/static")
)

# 2. Define the Modal App
app = modal.App("STEMagine-web")

# 3. Create the Flask App
flask_app = Flask(__name__)

# --- CONFIGURATION ---
# Service A: The Detector (DeBERTa)
DETECTOR_URL = "https://modal-labs-civicmachines--stemagine-classifier-llamaclas-489335.modal.run"

# Service B: The Rewriter (Llama-3)
REWRITER_URL = "https://modal-labs-civicmachines--fairframe-rewriter-rewriter-rewrite.modal.run"


@flask_app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@flask_app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main Orchestrator:
    1. Sends prompt to Detector.
    2. If gaps found -> Sends to Rewriter.
    3. Returns combined result to UI.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        # --- STEP 1: CALL DETECTOR (Service A) ---
        print(f"📡 Calling Detector: {DETECTOR_URL}")
        det_response = requests.post(DETECTOR_URL, json={"prompt": prompt})

        if not det_response.ok:
            raise Exception(f"Detector API Error: {det_response.status_code}")

        det_result = det_response.json()
        active_gaps = det_result.get("gaps", [])

        # --- STEP 2: CALL REWRITER (Service B) ---
        if active_gaps:
            print(f"⚡ Gaps found ({active_gaps}). Calling Rewriter: {REWRITER_URL}")

            payload = {
                "original_prompt": prompt,
                "gaps": active_gaps
            }

            rew_response = requests.post(REWRITER_URL, json=payload)

            if rew_response.ok:
                rew_data = rew_response.json()
                det_result["rewritten"] = rew_data.get("rewritten", "Error parsing rewrite.")
            else:
                print(f"❌ Rewriter Error: {rew_response.text}")
                det_result["rewritten"] = "Error: Could not generate rewrite."
        else:
            print("✅ Safe prompt. Skipping rewrite.")
            det_result["rewritten"] = None

        return jsonify(det_result)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({'error': str(e)}), 500


# --- DEPLOYMENT LOGIC ---
# This tells Modal: "Host this Flask app on the web"
# We explicitly 'mount' the templates and static folders so the cloud can see them.
@app.function(
    image=web_image
    # No mounts list needed here anymore
)
@modal.wsgi_app()
def flask_entrypoint():
    return flask_app


# Keep this for local testing (python app.py)
if __name__ == '__main__':
    print("🚀 Starting FairFrame locally...")
    flask_app.run(debug=True, host='0.0.0.0', port=5001)