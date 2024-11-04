const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { exec, execSync } = require('child_process');
const Excel = require('exceljs');
const moment = require('moment');
let mainWindow;
const dockerComposePath = path.join(__dirname, 'docker-compose.yml');

// Function to clean the orthanc_db directory
function cleanOrthancDb() {
    try {
        // Get the parent directory of the API (where main.js is located)
        const parentDir = path.join(__dirname, '..');
        const orthancDbPath = path.join(parentDir, 'orthanc_db');

        console.log('Starting Orthanc DB cleanup at:', orthancDbPath);

        // Check if directory exists
        if (!fs.existsSync(orthancDbPath)) {
            console.log('Orthanc DB directory does not exist:', orthancDbPath);
            return;
        }

        // List of files/directories to preserve
        const preserveList = ['WebViewerCache'];

        // Read all items in the directory
        const items = fs.readdirSync(orthancDbPath);

        for (const item of items) {
            const itemPath = path.join(orthancDbPath, item);

            // Check if the item should be preserved
            const shouldPreserve = preserveList.some(preserveItem => 
                item === preserveItem || item.startsWith(preserveItem + '.')
            );

            if (!shouldPreserve) {
                const stats = fs.statSync(itemPath);
                
                if (stats.isDirectory()) {
                    // Remove directory and its contents
                    fs.rmSync(itemPath, { recursive: true, force: true });
                    console.log(`Removed directory: ${itemPath}`);
                } else {
                    // Remove file
                    fs.unlinkSync(itemPath);
                    console.log(`Removed file: ${itemPath}`);
                }
            } else {
                console.log(`Preserved: ${itemPath}`);
            }
        }
        console.log('Orthanc DB cleanup completed successfully');
    } catch (error) {
        console.error('Error during Orthanc DB cleanup:', error);
        console.error('Error details:', error.stack);
    }
}

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

ipcMain.handle('open-questionnaire', () => {
    const questionnaireWindow = new BrowserWindow({
        width: 1000,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        }
    });
    
    questionnaireWindow.loadFile('src/questionnaire.html');
});


app.on('ready', createWindow);

ipcMain.handle('select-directory', async (event) => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
    });
    return result.filePaths[0];
});

// Helper function to validate directory and create if not exists
const validateDirectory = (dir, name, create = false) => {
    if (!dir) {
        throw new Error(`${name} directory not specified`);
    }
    
    if (!fs.existsSync(dir)) {
        if (create) {
            try {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`Created ${name} directory: ${dir}`);
            } catch (error) {
                throw new Error(`Failed to create ${name} directory: ${error.message}`);
            }
        } else {
            throw new Error(`${name} directory does not exist: ${dir}`);
        }
    }
    return dir.replace(/\\/g, '/');
};

const waitForContainer = (containerName) => {
    return new Promise((resolve) => {
        const interval = setInterval(() => {
            try {
                const result = execSync(`docker ps --filter "name=${containerName}" --format "{{.Status}}"`).toString().trim();
                if (!result) {
                    clearInterval(interval);
                    resolve();
                }
            } catch (error) {
                console.error(`Error checking ${containerName} status:`, error.message);
                clearInterval(interval);
                resolve();
            }
        }, 5000);
    });
};

ipcMain.handle('start-docker-compose', async (event, inputDir, outputDir, dicomOutputDir) => {
    try {
        // Validate input directory
        const validInputDir = validateDirectory(inputDir, 'Input');
        
        // Create and validate output directories
        const validOutputDir = validateDirectory(outputDir, 'Output', true);
        const validDicomOutputDir = validateDirectory(dicomOutputDir, 'DICOM output', true);
        
        // Create anonymized directory inside the output directory
        const anonymizedDir = path.join(outputDir, 'anonymized');
        const validAnonymizedDir = validateDirectory(anonymizedDir, 'Anonymized', true);

        // Create environment object with all directories
        const env = {
            ...process.env,
            INPUT_FOLDER: validInputDir,
            ANONYMIZED_FOLDER: validAnonymizedDir,
            OUTPUT_FOLDER: validOutputDir,
            DICOM_OUTPUT_FOLDER: validDicomOutputDir,
        };

        console.log('Starting with directories:', {
            INPUT_FOLDER: env.INPUT_FOLDER,
            ANONYMIZED_FOLDER: env.ANONYMIZED_FOLDER,
            OUTPUT_FOLDER: env.OUTPUT_FOLDER,
            DICOM_OUTPUT_FOLDER: env.DICOM_OUTPUT_FOLDER,
        });

        // Stop any running containers
        await new Promise((resolve) => {
            exec('docker-compose down', { 
                cwd: path.dirname(dockerComposePath),
                env: env
            }, (error) => {
                if (error) {
                    console.warn('Warning: Error during docker-compose down:', error.message);
                }
                resolve();
            });
        });

        // Start anonymizer
        console.log('Starting anonymization process...');
        await new Promise((resolve, reject) => {
            exec('docker-compose up -d dicom_anonymizer', {
                cwd: path.dirname(dockerComposePath),
                env: env
            }, (error, stdout, stderr) => {
                if (error) {
                    console.error('Error starting anonymizer:', stderr);
                    reject(error);
                } else {
                    console.log('Anonymizer started:', stdout);
                    resolve();
                }
            });
        });

        // Wait for anonymizer to complete
        console.log('Waiting for anonymization to complete...');
        await waitForContainer('anonymizer');
        console.log('Anonymization completed');

        // Verify anonymized files exist
        const anonymizedFiles = fs.readdirSync(validAnonymizedDir);
        if (anonymizedFiles.length === 0) {
            throw new Error('No anonymized files were created');
        }
        console.log(`Found ${anonymizedFiles.length} anonymized files`);

        // Start other services
        console.log('Starting remaining services...');
        await new Promise((resolve, reject) => {
            exec('docker-compose up -d zone_segmentation orthanc_server ohif_viewer visualization', {
                cwd: path.dirname(dockerComposePath),
                env: env
            }, (error, stdout, stderr) => {
                if (error) {
                    console.error('Error starting services:', stderr);
                    reject(error);
                } else {
                    console.log('Services started:', stdout);
                    resolve();
                }
            });
        });

        console.log('All services started successfully');

    } catch (error) {
        console.error('Error in pipeline:', error);
        throw error;
    }
});

// ... (rest of the code remains the same)

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
            }, 30000);
        }, 10000);
    });
});

ipcMain.handle('open-output-folder', (event, outputDir) => {
    shell.openPath(outputDir);
});

ipcMain.handle('open-ohif-viewer', () => {
    shell.openExternal('http://localhost:3000');
});

ipcMain.handle('open-anonymized-folder', (event, outputDir) => {
    const anonymizedDir = path.join(outputDir, 'anonymized');
    if (fs.existsSync(anonymizedDir)) {
        shell.openPath(anonymizedDir);
    } else {
        console.error('Anonymized directory does not exist:', anonymizedDir);
        // Optionally show an error dialog to the user
        dialog.showErrorBox(
            'Directory Not Found',
            'The anonymized data directory does not exist. Please run the tool first.'
        );
    }
});

ipcMain.handle('open-visualization', () => {
    shell.openExternal('http://localhost:8050');
});

ipcMain.handle('save-questionnaire', async (event, responses) => {
    try {
        const outputDir = validateDirectory(process.env.OUTPUT_FOLDER || './output', 'Output', true);
        const filePath = path.join(outputDir, 'questionnaire_responses.json');
        
        let questionnaires = [];
        
        // Check if file exists and read it
        if (fs.existsSync(filePath)) {
            try {
                const fileContent = fs.readFileSync(filePath, 'utf8');
                questionnaires = JSON.parse(fileContent);
                
                // Ensure questionnaires is an array even if file exists
                if (!Array.isArray(questionnaires)) {
                    questionnaires = [];
                }
            } catch (parseError) {
                console.error('Error parsing existing JSON:', parseError);
                questionnaires = [];
            }
        }

        // Create new response entry with timestamp
        const newResponse = {
            timestamp: moment().format('YYYY-MM-DD HH:mm:ss'),
            ...responses
        };

        // Add new response to array
        questionnaires.push(newResponse);

        // Write the updated array back to file
        fs.writeFileSync(filePath, JSON.stringify(questionnaires, null, 2), 'utf8');
        
        console.log('Questionnaire responses saved successfully to:', filePath);
        
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'Success',
            message: `Questionnaire responses saved successfully to ${filePath}`
        });
        
        return { success: true };
    } catch (error) {
        console.error('Error saving questionnaire responses:', error);
        dialog.showErrorBox(
            'Error',
            `Failed to save questionnaire: ${error.message}`
        );
        throw error;
    }
});

app.on('window-all-closed', () => {
    console.log('All windows closed, starting cleanup...');
    cleanOrthancDb();
    
    if (process.platform !== 'darwin') {
        app.quit();
    }
});