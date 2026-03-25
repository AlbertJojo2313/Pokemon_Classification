document.getElementById("pokemonForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("imageUpload");
    const resultDiv = document.getElementById("result");
    const pokemonName = document.getElementById("pokemonName");
    const pokemonConfidence = document.getElementById("pokemonConfidence");

    if (!fileInput.files[0]) {
        alert("⚠️ Please select an image first!");
        return;
    }

    resultDiv.style.display = "block";
    pokemonName.innerHTML = "🔍 Scanning...";
    pokemonConfidence.innerHTML = "";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        console.log("Status:", response.status);        // ✅ add this
        const data = await response.json();
        console.log("Response data:", data);            // ✅ add this

        pokemonName.innerHTML = `⚡ It's <span style="color:#ff0000;"><strong>${data.predicted}</strong></span>!`;
        pokemonConfidence.innerHTML = `Confidence: <strong>${data.confidence}%</strong>`;

    } catch (error) {
        console.error("Full error:", error);            // ✅ add this
        pokemonName.innerHTML = "❌ Error connecting to the Pokédex server!";
        pokemonConfidence.innerHTML = "";
    }
});