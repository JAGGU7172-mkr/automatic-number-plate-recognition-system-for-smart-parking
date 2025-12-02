document.addEventListener('DOMContentLoaded', function() {
    // Initialize DataTables for vehicle logs
    const vehicleLogsTable = document.getElementById('vehicle-logs-table');
    if (vehicleLogsTable) {
        const logsTable = new DataTable('#vehicle-logs-table', {
            order: [[0, 'desc']], // Sort by first column (timestamp) descending
            responsive: true,
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100],
            dom: 'frtip', // Removed 'B' to hide default buttons
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search logs...",
                lengthMenu: "Show _MENU_ logs per page",
                info: "Showing _START_ to _END_ of _TOTAL_ logs",
                infoEmpty: "No logs available",
                infoFiltered: "(filtered from _MAX_ total logs)",
                paginate: {
                    first: '<i class="fa fa-angle-double-left"></i>',
                    previous: '<i class="fa fa-angle-left"></i>',
                    next: '<i class="fa fa-angle-right"></i>',
                    last: '<i class="fa fa-angle-double-right"></i>'
                }
            }
        });
        
        // Connect custom export buttons
        const exportCSV = document.getElementById('exportCSV');
        if (exportCSV) {
            exportCSV.addEventListener('click', function() {
                // Create and trigger a CSV download
                const rows = Array.from(vehicleLogsTable.querySelectorAll('tbody tr'));
                const headers = Array.from(vehicleLogsTable.querySelectorAll('thead th'));
                
                // Find the index of the image column to exclude it
                const imageColumnIndex = headers.findIndex(th => 
                    th.textContent.trim().toLowerCase() === 'image');
                
                // Filter out the image column from headers
                const filteredHeaders = headers
                    .filter((_, index) => index !== imageColumnIndex)
                    .map(th => th.textContent.trim());
                
                let csvContent = filteredHeaders.join(',') + '\n';
                
                rows.forEach(row => {
                    const cells = Array.from(row.querySelectorAll('td'));
                    // Filter out the image column from row data
                    const rowData = cells
                        .filter((_, index) => index !== imageColumnIndex)
                        .map(cell => {
                            let text = cell.textContent.trim().replace(/\s+/g, ' ');
                            return '"' + text.replace(/"/g, '""') + '"';
                        });
                    csvContent += rowData.join(',') + '\n';
                });
                
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', 'vehicle-logs.csv');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
        }
        
        // PDF export handler
        document.getElementById('exportPDF').addEventListener('click', function() {
            // Get table data
            const rows = Array.from(vehicleLogsTable.querySelectorAll('tbody tr'));
            const headerCells = Array.from(vehicleLogsTable.querySelectorAll('thead th'));
            
            // Define document definition
            const docDefinition = {
                pageOrientation: 'landscape',
                pageMargins: [40, 60, 40, 60],
                header: {
                    columns: [
                        {
                            text: new Date().toLocaleString(),
                            alignment: 'left',
                            margin: [40, 20, 0, 0],
                            fontSize: 10,
                            color: '#666'
                        },
                        {
                            text: 'Vehicle Logs - ANPR Vehicle Tracking System',
                            alignment: 'right',
                            margin: [0, 20, 40, 0],
                            fontSize: 10,
                            bold: true
                        }
                    ]
                },
                content: [
                    {
                        text: 'ANPR System',
                        fontSize: 22,
                        bold: true,
                        alignment: 'center',
                        margin: [0, 0, 0, 20]
                    },
                    {
                        table: {
                            headerRows: 1,
                            widths: Array(headerCells.length).fill('*'),
                            body: [
                                // Header row
                                headerCells.map(cell => {
                                    return {
                                        text: cell.textContent.trim(),
                                        style: 'tableHeader'
                                    };
                                }),
                                // Data rows
                                ...rows.map(row => {
                                    const cells = Array.from(row.querySelectorAll('td'));
                                    return cells.map((cell, index) => {
                                        // Check for badges
                                        const badge = cell.querySelector('.badge');
                                        if (badge) {
                                            const text = badge.textContent.trim();
                                            let fillColor = null;
                                            
                                            if (badge.classList.contains('bg-success')) {
                                                fillColor = '#198754';
                                            } else if (badge.classList.contains('bg-danger')) {
                                                fillColor = '#dc3545';
                                            } else if (badge.classList.contains('bg-primary')) {
                                                fillColor = '#0d6efd';
                                            } else if (badge.classList.contains('bg-secondary')) {
                                                fillColor = '#6c757d';
                                            }
                                            
                                            return {
                                                text: text,
                                                fillColor: fillColor,
                                                color: 'white',
                                                fontSize: 10,
                                                alignment: 'center'
                                            };
                                        }
                                        
                                        // Check for image buttons
                                        const button = cell.querySelector('.btn');
                                        if (button && button.textContent.includes('View')) {
                                            return {
                                                text: 'Image Available',
                                                color: '#0dcaf0',
                                                italics: true,
                                                fontSize: 10
                                            };
                                        }
                                        
                                        // Regular cell
                                        return {
                                            text: cell.textContent.trim().replace(/\s+/g, ' '),
                                            fontSize: 10
                                        };
                                    });
                                })
                            ]
                        },
                        layout: {
                            hLineWidth: function(i, node) {
                                return (i === 0 || i === node.table.body.length) ? 2 : 1;
                            },
                            vLineWidth: function(i, node) {
                                return (i === 0 || i === node.table.widths.length) ? 2 : 1;
                            },
                            hLineColor: function(i, node) {
                                return (i === 0 || i === node.table.body.length) ? '#666' : '#ddd';
                            },
                            vLineColor: function(i, node) {
                                return (i === 0 || i === node.table.widths.length) ? '#666' : '#ddd';
                            }
                        }
                    }
                ],
                footer: function(currentPage, pageCount) {
                    return {
                        columns: [
                            {
                                text: 'ANPR Vehicle Tracking System - Confidential',
                                alignment: 'center',
                                fontSize: 8,
                                margin: [0, 10, 0, 0],
                                color: '#666'
                            },
                            {
                                text: 'Page ' + currentPage.toString() + ' of ' + pageCount,
                                alignment: 'right',
                                fontSize: 8,
                                margin: [0, 10, 40, 0],
                                color: '#666'
                            }
                        ]
                    };
                },
                styles: {
                    tableHeader: {
                        bold: true,
                        fontSize: 11,
                        color: 'white',
                        fillColor: '#343a40',
                        alignment: 'center'
                    }
                },
                defaultStyle: {
                    fontSize: 10,
                    color: '#333'
                }
            };
            
            // Generate and download PDF
            pdfMake.createPdf(docDefinition).download('vehicle-logs-report.pdf');
        });
        
        // Print function
        const printLogs = document.getElementById('printLogs');
        if (printLogs) {
            printLogs.addEventListener('click', function() {
                const table = document.getElementById('vehicle-logs-table');
                if (!table) return;
                
                // Create a new window for better print layout
                const printWindow = window.open('', '_blank');
                
                // Add print-friendly styles
                printWindow.document.write(`
                    <html>
                    <head>
                        <title>Vehicle Logs - Print View</title>
                        <style>
                            body { 
                                font-family: Arial, sans-serif; 
                                margin: 20px; 
                                background-color: white;
                                color: #333;
                            }
                            .header {
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                                margin-bottom: 10px;
                                border-bottom: 1px solid #ddd;
                                padding-bottom: 10px;
                            }
                            .date { 
                                font-size: 14px;
                                color: #333;
                            }
                            .title {
                                font-size: 16px;
                                font-weight: bold;
                                color: #333;
                            }
                            .subtitle {
                                display: flex;
                                align-items: center;
                                margin-bottom: 30px;
                                color: #777;
                                font-size: 24px;
                            }
                            .icon {
                                margin-right: 10px;
                            }
                            table { 
                                width: 100%; 
                                border-collapse: collapse; 
                                margin-bottom: 20px;
                            }
                            th, td { 
                                border: 1px solid #ddd; 
                                padding: 8px; 
                                text-align: left; 
                            }
                            th { 
                                background-color: #343a40; 
                                color: white; 
                            }
                            tr:nth-child(even) { 
                                background-color: #f9f9f9; 
                            }
                            tr:nth-child(odd) { 
                                background-color: #fff; 
                            }
                            .badge {
                                padding: 3px 6px;
                                border-radius: 4px;
                                font-weight: bold;
                                font-size: 11px;
                                display: inline-block;
                            }
                            .entry { 
                                background-color: #0d6efd;
                                color: white;
                            }
                            .exit { 
                                background-color: #6c757d;
                                color: white;
                            }
                            .authorized { 
                                background-color: #198754;
                                color: white;
                            }
                            .unauthorized { 
                                background-color: #dc3545;
                                color: white;
                            }
                            .not-allowed { 
                                background-color: #dc3545;
                                color: white;
                            }
                            .image-available {
                                color: #0dcaf0;
                                font-style: italic;
                            }
                            .footer { 
                                border-top: 1px solid #ddd; 
                                padding-top: 10px; 
                                margin-top: 20px; 
                                text-align: center; 
                                color: #777; 
                                font-size: 12px; 
                            }
                            @media print {
                                .no-print { display: none; }
                                body { 
                                    margin: 0; 
                                    padding: 15px;
                                }
                                th { 
                                    background-color: #343a40 !important; 
                                    color: white !important; 
                                    -webkit-print-color-adjust: exact !important;
                                    print-color-adjust: exact !important;
                                }
                                .badge {
                                    -webkit-print-color-adjust: exact !important;
                                    print-color-adjust: exact !important;
                                }
                            }
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <div class="date">${new Date().toLocaleDateString('en-US', { 
                                month: '2-digit', 
                                day: '2-digit', 
                                year: '2-digit' 
                            })}, ${new Date().toLocaleTimeString('en-US', {
                                hour: 'numeric',
                                minute: '2-digit',
                                hour12: true
                            })}</div>
                            <div class="title">Vehicle Logs - ANPR Vehicle Tracking System</div>
                        </div>
                        
                        <div class="subtitle">
                            <span class="icon">🚗</span>
                            <span>ANPR System</span>
                        </div>
                `);
                
                // Clone and modify the table for printing
                const tableClone = table.cloneNode(true);
                
                // Replace badges with styled spans
                const badges = tableClone.querySelectorAll('.badge');
                badges.forEach(badge => {
                    const text = badge.textContent.trim();
                    let className = '';
                    
                    if (badge.classList.contains('bg-success')) {
                        className = 'authorized';
                    } else if (badge.classList.contains('bg-danger')) {
                        className = 'unauthorized';
                    } else if (badge.classList.contains('bg-primary')) {
                        className = 'entry';
                    } else if (badge.classList.contains('bg-secondary')) {
                        className = 'exit';
                    } else {
                        className = 'not-allowed';
                    }
                    
                    badge.outerHTML = `<span class="badge ${className}">${text}</span>`;
                });
                
                // Replace image buttons with styled text
                const buttons = tableClone.querySelectorAll('.btn');
                buttons.forEach(button => {
                    button.outerHTML = '<span class="image-available">Image Available</span>';
                });
                
                printWindow.document.write(tableClone.outerHTML);
                
                // Add footer
                printWindow.document.write(`
                        <div class="footer">ANPR Vehicle Tracking System - Confidential</div>
                        <div class="no-print" style="text-align: center; margin-top: 20px;">
                            <button onclick="window.print()" style="background-color: #0d6efd; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Print</button>
                            <button onclick="window.close()" style="background-color: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-left: 8px;">Close</button>
                        </div>
                    </body>
                    </html>
                `);
                
                printWindow.document.close();
                
                // Trigger print after content is loaded
                printWindow.onload = function() {
                    setTimeout(function() {
                        printWindow.focus();
                    }, 500);
                };
            });
        }
    }
});
