(function() {
    let activityChartInstance = null;
    let vehicleStatusChartInstance = null;
    let isInitialized = false;

    function initializeCharts() {
        if (isInitialized) return;
        isInitialized = true;
        
        initActivityChart();
        initVehicleStatusChart();
        updateTime();
    }

    document.addEventListener('DOMContentLoaded', initializeCharts);

    function initActivityChart() {
        const activityChartEl = document.getElementById('activityChart');
        if (!activityChartEl) return;
        
        let dateLabels = [];
        let entryData = [];
        let exitData = [];
        
        try {
            // Log raw data for debugging
            console.log("Raw data attributes:");
            console.log("Labels:", activityChartEl.dataset.labels);
            console.log("Entries:", activityChartEl.dataset.entries);
            console.log("Exits:", activityChartEl.dataset.exits);
            
            // More robust parsing with error handling
            try {
                dateLabels = JSON.parse(activityChartEl.dataset.labels || '[]');
            } catch (e) {
                console.error("Error parsing labels:", e);
                dateLabels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"];
            }
            
            try {
                entryData = JSON.parse(activityChartEl.dataset.entries || '[]');
            } catch (e) {
                console.error("Error parsing entries:", e);
                entryData = Array(24).fill(0);
            }
            
            try {
                exitData = JSON.parse(activityChartEl.dataset.exits || '[]');
            } catch (e) {
                console.error("Error parsing exits:", e);
                exitData = Array(24).fill(0);
            }
            
            console.log("Parsed data:");
            console.log("Labels:", dateLabels);
            console.log("Entries:", entryData);
            console.log("Exits:", exitData);
        } catch (error) {
            console.error("General error in chart data processing:", error);
            // Provide default values if parsing fails
            dateLabels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"];
            entryData = Array(24).fill(0);
            exitData = Array(24).fill(0);
        }
        
        // Calculate max value for y-axis with a minimum of 1 and add padding
        const maxValue = Math.max(...entryData, ...exitData);
        const maxCount = maxValue > 0 ? maxValue + 1 : 1;
        console.log("Max count for chart:", maxCount);
    
        if (activityChartInstance) {
            activityChartInstance.destroy();
        }
        
        // Force chart to be visible before initialization
        activityChartEl.style.display = 'block';
        
        activityChartInstance = new Chart(activityChartEl, {
            type: 'bar',
            data: {
                labels: dateLabels,
                datasets: [
                    {
                        label: 'Entries',
                        data: entryData,
                        backgroundColor: 'rgba(40, 167, 69, 0.5)',  // Green color to match the legend
                        borderColor: '#28a745',  // Green border
                        borderWidth: 1,
                        borderRadius: 5,
                        barPercentage: 0.6,
                        categoryPercentage: 0.8
                    },
                    {
                        label: 'Exits',
                        data: exitData,
                        backgroundColor: 'rgba(255, 193, 7, 0.5)',  // Yellow color (unchanged)
                        borderColor: '#ffc107',  // Yellow border (unchanged)
                        borderWidth: 1,
                        borderRadius: 5,
                        barPercentage: 0.6,
                        categoryPercentage: 0.8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: maxCount,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#fff',
                            font: { size: 11 },
                            stepSize: 1
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#fff',
                            font: { size: 11 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false,  // Hide the built-in legend
                        position: 'top',
                        labels: {
                            color: '#fff'
                        }
                    }
                }
            }
        });
        
        console.log("Activity chart initialized");
    }

    function initVehicleStatusChart() {
        const vehicleStatusEl = document.getElementById('vehicleStatusChart');
        if (!vehicleStatusEl) {
            console.error("Vehicle status chart element not found");
            return;
        }
        
        let authorized = 0;
        let unauthorized = 0;
        
        try {
            authorized = vehicleStatusEl.dataset.authorized ? parseInt(vehicleStatusEl.dataset.authorized) || 0 : 0;
            unauthorized = vehicleStatusEl.dataset.unauthorized ? parseInt(vehicleStatusEl.dataset.unauthorized) || 0 : 0;
        } catch (error) {
            console.error("Error parsing vehicle status data:", error);
        }
        
        if (vehicleStatusChartInstance) {
            vehicleStatusChartInstance.destroy();
        }
        
        vehicleStatusChartInstance = new Chart(vehicleStatusEl, {
            type: 'doughnut',
            data: {
                labels: ['Authorized', 'Unauthorized'],
                datasets: [{
                    data: [authorized, unauthorized],
                    backgroundColor: ['#20c997', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            padding: 40,
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });
    }

    function updateTime() {
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = now.toLocaleTimeString();
        }
        
        // Update the date with correct timezone
        const dateElement = document.querySelector('.d-flex.justify-content-between.align-items-center .text-muted');
        if (dateElement) {
            const now = new Date();
            const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
            dateElement.innerHTML = '<i class="fas fa-calendar-alt me-1"></i> ' + now.toLocaleDateString('en-US', options);
        }
    }

    setInterval(updateTime, 1000);
})();