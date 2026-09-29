document.addEventListener('DOMContentLoaded', () => {
    const reviewInput = document.getElementById('reviewText');
    const checkBtn = document.getElementById('checkBtn');
    const resultCard = document.getElementById('resultCard');
    const resultBadge = document.getElementById('resultBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const errorAlert = document.getElementById('errorAlert');

    // Realistic example reviews matching dataset patterns
    const exampleReviews = {
        ex1: "These are just perfect, exactly what I was looking for. Assembly was simple and shipping was fast.",
        ex2: "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty, great product for the price!",
        ex3: "What can you say--- cheap and it works as intended. I've had it for two months now without any issues."
    };

    // Attach click handlers to example buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const exKey = e.target.getAttribute('data-example');
            if (exampleReviews[exKey]) {
                reviewInput.value = exampleReviews[exKey];
                hideResults();
                reviewInput.focus();
            }
        });
    });

    function hideResults() {
        if (resultCard) resultCard.classList.remove('show');
        if (errorAlert) errorAlert.style.display = 'none';
    }

    if (checkBtn) {
        checkBtn.addEventListener('click', async () => {
            const text = reviewInput.value.trim();

            hideResults();

            if (!text) {
                showError("Please enter a review before clicking Check Review.");
                return;
            }

            // Disable button during API request
            checkBtn.disabled = true;
            checkBtn.innerText = "Analyzing...";

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (!response.ok) {
                    showError(data.error || "An error occurred while analyzing the review.");
                } else {
                    displayResult(data);
                }
            } catch (err) {
                showError("Failed to connect to the prediction server. Please ensure Flask is running.");
                console.error("Prediction error:", err);
            } finally {
                checkBtn.disabled = false;
                checkBtn.innerText = "Check Review";
            }
        });
    }

    function displayResult(data) {
        if (!resultCard || !resultBadge || !confidenceValue) return;

        resultBadge.textContent = data.label;

        if (data.is_genuine) {
            resultBadge.className = 'badge badge-genuine';
        } else {
            resultBadge.className = 'badge badge-fake';
        }

        confidenceValue.textContent = `${data.confidence}%`;
        resultCard.classList.add('show');
    }

    function showError(msg) {
        if (errorAlert) {
            errorAlert.textContent = msg;
            errorAlert.style.display = 'block';
        }
    }
});
