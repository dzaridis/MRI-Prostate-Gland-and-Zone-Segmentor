const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const { exec } = require('child_process');

let mainWindow;

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
    const dockerComposePath = path.join(__dirname, '../docker-compose.yml');
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
