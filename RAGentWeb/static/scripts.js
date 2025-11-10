// Base URL for API calls
const BASE_API_URL = ""; 

document.addEventListener('DOMContentLoaded', () => {

    // 1. Tab Switching Logic
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    function showTab(tabId) {
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        const activeTabButton = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
        const activeTabContent = document.getElementById(tabId);
        
        if (activeTabButton) activeTabButton.classList.add('active');
        if (activeTabContent) activeTabContent.classList.add('active');
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => showTab(button.dataset.tab));
    });

    // Initialize first tab
    showTab('ml');

    // 2. Utility Functions
    function setOutput(tabId, content) {
        const outputArea = document.querySelector(`#${tabId}_output`);
        if (outputArea) {
            outputArea.innerHTML = content;
        }
    }

    function startLoading(button) {
        button.disabled = true;
        button.originalText = button.textContent;
        button.textContent = "Processing... Please wait.";
    }

    function stopLoading(button) {
        button.disabled = false;
        button.textContent = button.originalText;
    }

    function formatWeatherData(data) {
        let html = '<h4>Weather Data Used:</h4><ul class="list-disc ml-5">';
        for (const [key, value] of Object.entries(data)) {
            html += `<li><strong>${key.replace('_', ' ')}:</strong> ${value}</li>`;
        }
        html += '</ul>';
        return html;
    }

    // 3. ML Prediction Logic
    const mlForm = document.getElementById('ml-form');
    if (mlForm) {
        mlForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const age = document.getElementById('ml-age').value.trim();
            const gender = document.getElementById('ml-gender').value.trim();
            const city = document.getElementById('ml-city').value.trim();
            const symptoms = document.getElementById('ml-symptoms').value.trim();
            const submitBtn = mlForm.querySelector('button[type="submit"]');

            if (!age || !gender || !city || !symptoms) {
                setOutput('ml', '<h3 class="text-red-600">All fields are required.</h3>');
                return;
            }

            startLoading(submitBtn);
            setOutput('ml', '<h3>Processing...</h3><p>Fetching weather data and running ML model.</p>');

            try {
                const response = await fetch(`${BASE_API_URL}/api/predict_ml`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ age, gender, city, symptoms })
                });

                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to get ML prediction.');

                const confidence = (result.confidence * 100).toFixed(2);
                const weatherDataHtml = result.raw_weather_data ? formatWeatherData(result.raw_weather_data) : '';

                setOutput('ml', `
                    <h3>ML Prediction Result</h3>
                    <p><strong>Status:</strong> ${result.status}</p>
                    <p><strong>Predicted Risk:</strong> <span class="text-xl font-bold text-green-700">${result.prediction}</span></p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <p><strong>City:</strong> ${city}</p>
                    <p><strong>Age:</strong> ${age}</p>
                    <p><strong>Gender:</strong> ${gender}</p>
                    <p><strong>Symptoms:</strong> ${symptoms}</p>
                    ${weatherDataHtml}
                `);

            } catch (error) {
                setOutput('ml', `<h3 class="text-red-600">Error:</h3><p>${error.message}</p>`);
            } finally {
                stopLoading(submitBtn);
            }
        });
    }

    // 4. DL Prediction Logic
    const dlForm = document.getElementById('dl-form');
    if (dlForm) {
        dlForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const note = document.getElementById('dl-note').value.trim();
            const submitBtn = dlForm.querySelector('button[type="submit"]');

            if (!note) {
                setOutput('dl', '<h3 class="text-red-600">Note is required.</h3>');
                return;
            }

            startLoading(submitBtn);
            setOutput('dl', '<h3>Processing...</h3><p>Running DL model.</p>');

            try {
                const response = await fetch(`${BASE_API_URL}/api/predict_dl`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ note })
                });

                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to get DL prediction.');

                const confidence = (result.confidence * 100).toFixed(2);
                const weatherDataHtml = result.raw_weather_data ? formatWeatherData(result.raw_weather_data) : '';

                setOutput('dl', `
                    <h3>DL Prediction Result</h3>
                    <p><strong>Status:</strong> ${result.status}</p>
                    <p><strong>Predicted Risk:</strong> <span class="text-xl font-bold text-blue-700">${result.prediction}</span></p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <p><strong>User Note:</strong> ${note}</p>
                    ${weatherDataHtml}
                `);

            } catch (error) {
                setOutput('dl', `<h3 class="text-red-600">Error:</h3><p>${error.message}</p>`);
            } finally {
                stopLoading(submitBtn);
            }
        });
    }

    // 5. Image Classification Logic

    const imageForm = document.getElementById('image-form');
    if (imageForm) {
        imageForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('image-file');
            const submitBtn = imageForm.querySelector('button[type="submit"]');

            if (fileInput.files.length === 0) {
                setOutput('image', '<h3>Error</h3><p class="text-red-600">Please select an image file.</p>');
                return;
            }

            startLoading(submitBtn);
            setOutput('image', '<h3>Processing...</h3><p>Uploading and running Image Classification model.</p>');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch(`${BASE_API_URL}/api/classify_image`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.detail || 'Failed to classify image.');
                }
                
                let topPredictionsHtml = '<ul>';
                result.top_predictions.forEach(p => {
                    topPredictionsHtml += `<li>${p.label}: ${(p.confidence * 100).toFixed(2)}%</li>`;
                });
                topPredictionsHtml += '</ul>';

                setOutput('image', `
                    <h3>Image Classification Result</h3>
                    <p><strong>Status:</strong> ${result.status}</p>
                    <p><strong>Predicted Class:</strong> <span class="text-xl font-bold text-purple-700">${result.prediction}</span></p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                    <h4>Top 3 Predictions:</h4>
                    ${topPredictionsHtml}
                    `);

            } catch (error) {
                console.error("Image Classification Error:", error);
                setOutput('image', `<h3>Error</h3><p class="text-red-600">An error occurred: ${error.message}</p>`);
            } finally {
                stopLoading(submitBtn);
            }
        });
    }

    // 6. RAG Chat Logic
    const ragForm = document.getElementById('rag-form');
    const chatInput = document.getElementById('rag-query');
    const chatWindow = document.getElementById('chat-window');

    if (ragForm) {
        ragForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = chatInput.value.trim();
            const submitBtn = ragForm.querySelector('button[type="submit"]');

            if (!query) return;

            const userMsg = document.createElement('div');
            userMsg.className = 'chat-message user-message bg-gray-200 p-2 rounded';
            userMsg.textContent = query;
            chatWindow.appendChild(userMsg);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            chatInput.value = '';

            const aiLoadingMsg = document.createElement('div');
            aiLoadingMsg.className = 'chat-message ai-message bg-green-100 p-2 rounded';
            aiLoadingMsg.innerHTML = 'Thinking...';
            chatWindow.appendChild(aiLoadingMsg);
            chatWindow.scrollTop = chatWindow.scrollHeight;

            startLoading(submitBtn);

            try {
                const response = await fetch(`${BASE_API_URL}/api/rag_chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });

                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to get RAG response.');

                aiLoadingMsg.textContent = result.response;

                if (result.sources && typeof result.sources === 'string') {
                    let sourcesHtml = `<div class="mt-2 text-sm text-gray-600"><strong>Source Information:</strong> ${result.sources}</div>`;
                    aiLoadingMsg.innerHTML += sourcesHtml;
                }

            } catch (error) {
                aiLoadingMsg.innerHTML = `<span class="text-red-600">Error: ${error.message}</span>`;
            } finally {
                stopLoading(submitBtn);
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        });
    }


});
