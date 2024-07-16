const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { exec, execSync } = require('child_process');

let mainWindow;
const dockerComposePath = path.join(__dirname, 'docker-compose.yml');

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    mainWindow.loadFile('src/index.html');
}

app.on('ready', createWindow);

ipcMain.handle('select-directory', async (event) => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
    });
    return result.filePaths[0];
});

ipcMain.handle('start-docker-compose', (event, inputDir, outputDir, dicomOutputDir) => {
    const env = {
        ...process.env,
        INPUT_FOLDER: inputDir.replace(/\\/g, '/'),
        OUTPUT_FOLDER: outputDir.replace(/\\/g, '/'),
        DICOM_OUTPUT_FOLDER: dicomOutputDir.replace(/\\/g, '/'),
    };

    exec('docker-compose up -d', { cwd: path.dirname(dockerComposePath), env }, (err, stdout, stderr) => {
        if (err) {
            console.error(`Error starting Docker Compose: ${stderr}`);
        } else {
            console.log(`Docker Compose started: ${stdout}`);
        }
    });
});

ipcMain.handle('wait-for-segmentation', (event) => {
    return new Promise((resolve) => {
        setTimeout(() => {
            const interval = setInterval(() => {
                try {
                    const result = execSync('docker ps --filter "name=zone_segment" --format "{{.Status}}"').toString().trim();
                    if (!result) {
                        clearInterval(interval);
                        resolve();
                    }
                } catch (error) {
                    console.error(`Error checking container status: ${error.message}`);
                    clearInterval(interval);
                    resolve();
                }
            }, 30000); // Check every 30 seconds
        }, 10000); // Initial delay of 10 seconds
    });
});

ipcMain.handle('open-output-folder', (event, outputDir) => {
    shell.openPath(outputDir);
});

ipcMain.handle('open-ohif-viewer', () => {
    shell.openExternal('http://localhost:3000');
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
