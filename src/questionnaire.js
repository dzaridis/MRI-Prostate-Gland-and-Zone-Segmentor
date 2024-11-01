const { ipcRenderer } = require('electron');

document.getElementById('questionnaireForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const responses = {};
    
    formData.forEach((value, key) => {
        responses[key] = value;
    });
    
    try {
        await ipcRenderer.invoke('save-questionnaire', responses);
        alert('Responses saved successfully!');
        window.close();
    } catch (error) {
        alert('Error saving responses: ' + error.message);
    }
});