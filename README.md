# Deeplasia

A lightweight demo with a Dockerized backend and a simple frontend for single-image inference via web UI or CLI.
## 0. Resource Download
Download the system from GitHub. Since the model file is too big, we put the model file on google drive
Please use the file in

    https://drive.google.com/drive/folders/1F3N_WM2dU6VR4YLUfvTJHRhwD3IBFM0K?usp=sharing
and

    https://drive.google.com/drive/folders/1BLp1M5Nwqo_yFI9_8GjM1GBxX8r0uXfm?usp=sharing
        
to replace the file in

    capstone-project-25t2-9900-h16a-dount/Deeplasia-master/output/debug/version_0
and

    capstone-project-25t2-9900-h16a-dount/Deeplasia-master/output/debug/version_1
## 1. Environment Setup
1.1Navigate to the project directory 

    cd Deeplasia-master

1.2Ensure Docker is running in the background

Please make sure the Docker is installed and Docker daemon is active 	before proceeding.
## 2.Backend Deployment
2.1Build the backend image

    docker build -t deeplasia-backend . 

2.2Run the backend container

    docker run -p 5001:5000 deeplasia-backend
      
This command maps the container’s port 5000 to the host’s port 5001.Once the container starts successfully, the backend API will be available at http//localhost::5001.

## 3.Frontend Usage
### A.Web GUI

1.return to the project directory. 

2.Navigate to the frontend folder and locate the file: frontend/web.html.

3.Double-click web.html to open it in a web browser.

4.The web interface is designed face to general users, like doctors and customer service staff. It’s user friendly but currently only support single image upload.

5.Follow the clearly interface in webpage to upload files, choose gender and download.

### B.Command Line Interface (CLI)

1.Open a new terminal window.

2.Navigate to the frontend directory: 

    cd frontend
      
3.Run the CLI tool : 

    python cli.py --image path/to/image.jpg --male 1

  Replace the “path/to/image.jpg -” to your real image file path
  
  Set --male to 1 for male or 0 for female.
  
4.The generated reports will be saved in frontend/downloaded
And save path will showed in the terminal.

### C.Batch Prediction
1.Open a new terminal window.

2.Naigate to the frontend directory: 

      cd frontend
3.Run the batch prediction script:
   
      bash batch_predict.sh
      
4.By default, the script processes all images in: frontend/ test
To change the input folder, modify the IMAGE_DIR variable in batch_predict.sh. Or alternatively, place your test files directly into the test folder under frontend and re-run the command.

5.All report genarated will be automatically place to 	frontend/downloaded.

## 4.Notes
·  This installation guide is based on the current project setup.
·  Any updates to the process will be documented in the GitHub 		   README.

