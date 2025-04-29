# Import the server module
import uvicorn
from main import app

if __name__ == "__main__":
    # Start the uvicorn server
    uvicorn.run(
	  app, 
      # Configure the host
      host="0.0.0.0",
      # Configure the port
      port=8080)
