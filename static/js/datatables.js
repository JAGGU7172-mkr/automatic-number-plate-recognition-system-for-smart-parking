// DataTables initialization for vehicle logs

document.addEventListener('DOMContentLoaded', function() {
    // Vehicle logs table initialization moved to vehicle_logs.js
    
    // Initialize DataTables for vehicles table
    const vehiclesTable = document.getElementById('vehicles-table');
    if (vehiclesTable) {
        new DataTable('#vehicles-table', {
            order: [[5, 'desc']], // Sort by created date descending
            responsive: true,
            pageLength: 10,
            lengthMenu: [10, 25, 50, 100],
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search vehicles...",
                lengthMenu: "Show _MENU_ vehicles per page",
                info: "Showing _START_ to _END_ of _TOTAL_ vehicles",
                infoEmpty: "No vehicles available",
                infoFiltered: "(filtered from _MAX_ total vehicles)",
                paginate: {
                    first: '<i class="fa fa-angle-double-left"></i>',
                    previous: '<i class="fa fa-angle-left"></i>',
                    next: '<i class="fa fa-angle-right"></i>',
                    last: '<i class="fa fa-angle-double-right"></i>'
                }
            }
        });
    }

    // Initialize DataTables for users table
    const usersTable = document.getElementById('users-table');
    if (usersTable) {
        new DataTable('#users-table', {
            order: [[3, 'desc']], // Sort by created date descending
            responsive: true,
            pageLength: 10,
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search users...",
                lengthMenu: "Show _MENU_ users per page",
                info: "Showing _START_ to _END_ of _TOTAL_ users",
                infoEmpty: "No users available",
                infoFiltered: "(filtered from _MAX_ total users)",
                paginate: {
                    first: '<i class="fa fa-angle-double-left"></i>',
                    previous: '<i class="fa fa-angle-left"></i>',
                    next: '<i class="fa fa-angle-right"></i>',
                    last: '<i class="fa fa-angle-double-right"></i>'
                }
            }
        });
    }
});
