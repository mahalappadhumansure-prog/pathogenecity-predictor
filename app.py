
from flask import Flask, Response, jsonify, render_template, request

import model

app = Flask(__name__)


@app.get("/")
def index():
    """Display the main HBB Predictor webpage."""
    return render_template("index.html")


@app.get("/api/examples")
def examples():
    """Return example DNA sequences."""
    return jsonify(model.EXAMPLES)


@app.get("/api/model-info")
def model_info():
    """Return model evaluation metrics and global feature importance."""
    return jsonify({
        "metrics": model.METRICS,
        "global_importance": model.GLOBAL_IMPORTANCE
    })


@app.get("/summary.png")
def summary():
    """Return the SHAP summary plot."""
    return Response(
        model.SUMMARY_PNG,
        mimetype="image/png"
    )


@app.post("/api/predict")
def predict():
    """Analyze the submitted DNA sequence."""
    data = request.get_json(silent=True) or {}
    sequence = data.get("sequence", "")

    if not isinstance(sequence, str):
        return jsonify({
            "error": "DNA sequence must be provided as text."
        }), 400

    try:
        result = model.analyze(sequence)

        # Ensure the frontend receives the complete sequence.
        result["sequence"] = result.get("sequence", "")
        result["length"] = len(result["sequence"])

        # No Variant must not contain variant-position information.
        if result.get("variant") is None:
            result["variant"] = None

        # Ensure SHAP contributions are available as a list.
        if result.get("shap") is None:
            result["shap"] = []

        return jsonify(result)

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Unexpected error during HBB prediction")
        return jsonify({
            "error": (
                "An unexpected prediction error occurred. "
                "Check the Flask terminal for details."
            )
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
