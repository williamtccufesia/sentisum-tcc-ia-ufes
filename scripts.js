document.getElementById('searchForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Impede o envio padrão do formulário

    const searchQuery = document.getElementById('searchQuery').value;

    // Envia a consulta para o servidor
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ searchQuery }),
    })
    .then(response => response.json())
    .then(data => {
        // Exibe os resultados na página
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = ''; // Limpa resultados anteriores

        data.forEach(video => {
            const videoElement = document.createElement('div');
            videoElement.innerHTML = `
                <h2>${video.title}</h2>
                <p>${video.description}</p>
            `;
            resultsDiv.appendChild(videoElement);
        });
    });
});