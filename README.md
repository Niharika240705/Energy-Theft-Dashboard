Energy Theft Detection Dashboard
Overview:
The Energy Theft Detection Dashboard project provides an interactive platform for monitoring and identifying possible instances of electricity theft using smart meter data and advanced analytics. Designed for utility providers and researchers, the dashboard assists in analyzing consumption patterns, detecting anomalies, and visualizing suspected theft events.

Features:
1. Real-Time Data Monitoring: See live or periodic energy consumption data from smart meters and sensors.
2. Anomaly Detection: Detect suspicious usage and identify theft using machine learning models and statistical analysis.
3. Visualization Dashboard: Intuitive charts and historical logs of theft events.
4. Alert Notification: Configurable alert system to notify when theft is suspected.
5. Consumption Analytics: Deep analysis of usage (e.g., peak loads, appliance-level patterns).
6. User Management: Manage users and permissions for accessing the dashboard.

Technologies Used:
1. Frontend: Python Dash / Streamlit / React.js (Confirm the actual stack)
2. Backend: Python (pandas, scikit-learn, Flask or FastAPI)
3. Database: SQLite / PostgreSQL / MongoDB
4. Data Models: Machine Learning algorithms (Isolation Forest, Clustering, LSTM, SVM)
5. IoT Hardware: ESP32, ZMPT101B, ACS712 sensors
6. Visualization: Plotly, Blynk IoT integration

Project Structure:
Energy-Theft-Detection-Dashboard/
├── data/        # Sample datasets and logs
├── models/      # Trained ML models
├── src/         # Source code for dashboard and backend
├── scripts/     # Utility and data preparation scripts
├── images/      # Images for documentation
├── dashboards/  # Dashboard config files
├── requirements.txt
└── README.md

Setup and Installation
Prerequisites:

1. Python 3.x
2. Node.js (if using React frontend)
3. Sensor hardware (if real-time monitoring)
4. Blynk IoT app

Installation Steps

1. Clone Repository:
   git clone https://github.com/Niharika240705/Energy-Theft-Detection-Dashboard.git
   cd Energy-Theft-Detection-Dashboard

2. Install Dependencies:
   pip install -r requirements.txt

3. Configure Hardware:
   Connect ESP32 and sensors as per the documentation.
   Upload data or connect to data streaming source.
   
4. Run the Dashboard:
   streamlit run dashboard/app.py

Usage:
1. Access the dashboard locally via your browser.
2. Select time frames, view sensor logs, and analyze flagged theft events.
3. Configure alert and notification preferences.
4. Export analysis results.

Contributing:
Pull requests and feature suggestions are welcome. For major changes, please open an issue first to discuss what you would like to change.

License:
MIT License (or specify your license)




   
