// Spam Check Function
async function checkSpam() {
  const text = document.getElementById('emailInput').value.trim();
  const resultDiv = document.getElementById('result');
  const loader = document.getElementById('loader');

  resultDiv.innerHTML = "";
  resultDiv.className = "text-lg font-bold text-center mt-6";

  if (!text) {
    resultDiv.innerHTML = "⚠️ Please enter some text.";
    resultDiv.classList.add("text-yellow-500");
    return;
  }

  loader.classList.remove("hidden");
  resultDiv.classList.add("hidden");

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    let resultText;
    if (data.label === "SPAM") {
      resultText = `🚨 <span class="text-red-600">${data.label}</span> (Confidence: ${data.confidence}%)`;
    } else {
      resultText = `✅ <span class="text-green-600">${data.label}</span> (Confidence: ${data.confidence}%)`;
    }

    resultDiv.innerHTML = resultText;
    resultDiv.classList.remove("hidden");
  } catch (err) {
    resultDiv.innerHTML = "❌ Error contacting server.";
    resultDiv.classList.remove("hidden");
    resultDiv.classList.add("text-red-500");
  } finally {
    loader.classList.add("hidden");
  }
}

// Clear Input and Output
function clearFields() {
  document.getElementById('emailInput').value = '';
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '';
  resultDiv.className = "text-lg font-bold text-center mt-6";
}


