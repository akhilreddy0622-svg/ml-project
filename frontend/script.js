const btn = document.getElementById("analyzeBtn");
const box = document.getElementById("textInput");
const result = document.getElementById("result");
const loading = document.getElementById("loading");

btn.onclick = async () => {

  result.className = "result hidden";
  loading.classList.remove("hidden");

  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text: box.value})
  });

  const data = await res.json();

  loading.classList.add("hidden");

  if(!data.success){
    alert(data.message);
    return;
  }

  result.classList.remove("hidden");
  result.classList.add(data.sentiment);

  result.innerHTML =
    data.sentiment.toUpperCase() +
    "<br>" +
    (data.confidence*100).toFixed(1) + "% confidence";
};