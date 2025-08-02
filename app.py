from flask import Flask, request, jsonify, render_template
from recommend import load_embeddings, generate_recommendations

app = Flask(__name__)

hybrid_embeddings, user_hybrid_vectors = load_embeddings()


@app.route('/', methods=["GET", "POST"])
def home():
    sample_user_ids = [3875557, 1974976, 1739125, 2394239, 1696731]

    if request.method == "POST":
        user_id = int(request.form.get("user_id", 3875557))
        top_n = int(request.form.get("top_n", 20))

        recommendations = generate_recommendations(user_id, hybrid_embeddings, user_hybrid_vectors, top_n)
        if recommendations is None:
            return render_template("index.html", error="User not found.", sample_user_ids=sample_user_ids)

        return render_template("index.html", recs=recommendations, user_id=user_id, sample_user_ids=sample_user_ids)

    return render_template("index.html", sample_user_ids=sample_user_ids)

@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    try:
        user_id = int(request.args.get("user_id"))
        top_n = int(request.args.get("top_n", 20))

        recommendations = generate_recommendations(user_id, hybrid_embeddings, user_hybrid_vectors, top_n)

        if recommendations is None:
            return jsonify({"error": "User not found historical profile - do demographic preference"}), 404

        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)