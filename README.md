Execution Steps:<br>
1. Install your dependencies from requirements.txt, this would be run as: <br>
   1. Run: pip install -r requirements.txt
2. Train the model by running main.py and save that housing_price_model.h5 <br>
   1. Run: python main.py
3. Start the FastAPI Server
   1. Run: uvicorn app:app --reload
4. Test the API
   1. Use some sort of tool like Postman or curl to send a POST request with housing data from #2 to the /predict endpoint
5. Finally, main.py should visualize your results and show a loss plot during training