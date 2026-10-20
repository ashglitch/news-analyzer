
import os

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)

from analyzer import analyze_article
from database import (
    init_db, save_analysis,
    get_history, get_analysis, delete_analysis
)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "local-development-key-change-before-deploy"
)

init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    headline = request.form.get("headline", "").strip()
    article_text = request.form.get("article_text", "").strip()

    if not headline or not article_text:
        flash("Please enter both a headline and article text.", "error")
        return redirect(url_for("index"))

    try:
        result = analyze_article(headline, article_text)
        analysis_id = save_analysis(result)

        return redirect(url_for("results", analysis_id=analysis_id))

    except ValueError as error:
        flash(str(error), "error")
        return redirect(url_for("index"))

    except Exception:
        app.logger.exception("Article analysis failed.")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("index"))


@app.route("/results/<int:analysis_id>")
def results(analysis_id):
    saved = get_analysis(analysis_id)

    if saved is None:
        flash("That analysis could not be found.", "error")
        return redirect(url_for("index"))

    result = analyze_article(
        saved["headline"],
        saved["article_text"]
    )
    result["id"] = saved["id"]
    result["created_at"] = saved["created_at"]

    return render_template("results.html", result=result)


@app.route("/history")
def history():
    analyses = get_history()
    return render_template("history.html", analyses=analyses)


@app.route("/history/delete/<int:analysis_id>", methods=["POST"])
def delete(analysis_id):
    delete_analysis(analysis_id)
    flash("Analysis deleted.", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
