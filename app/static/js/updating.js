/**
 * Real-time updating code
 * 
 * We use Pusher for real-time updating for new proposals/awards
 */
var pusher = new Pusher('fa85ba9b05e4fe72c78f', {
    cluster: 'us2'
});

var awdChannel = pusher.subscribe('awards-channel');
var propChannel = pusher.subscribe('proposals-channel');

awdChannel.bind('award-updates', function(data) {
    addAwdRow(data)
});

propChannel.bind('proposal-updates', function(data) {
    addPropRow(data)
});

/**
 * UI flow
 * 
 * Instructions for buttons, forms, etc
 */
$('.custom-file-input').change(function(e) {
    var files = [];
    for (var i = 0; i < $(this)[0].files.length; i++) {
        files.push($(this)[0].files[i].name);
    }
    $(this).next('.custom-file-label').html(files.join(', '));
});

$('#uploadform').submit(
  function( e ) {
    msg = document.getElementById("errormsg");
    msg.style.display = 'none';

    btn = document.getElementById("checkcsv");
    btn.classList.add('disabled');
    btn.disabled = true
    btn.type = 'button';

    addLoadingCircle('maincontainer');

    $.ajax( {
        url: '/update',
        type: 'POST',
        data: new FormData( this ),
        processData: false,
        contentType: false,
        success: function(result) {
            //console.log(result);
            document.getElementById('loading-circle').remove()

            if (result.includes('Continue with next step')) { // if the csv's are valid change alert message to green
                msg.classList.remove('alert-danger')
                msg.classList.add('alert-success')

                console.log('adding update button')
                insertDoUpdateButton('maincontainer')
            } else { // if the csv's are invalid, re-enable the button
                btn.classList.remove('disabled')
                btn.type = 'submit'
            }

            msg.style.display = 'block';
            $('#errormsg').text(result);
        }
    });

    e.preventDefault();
  } 
);

$(document).on('click','#checknewgrants', function(e) {
    btn = document.getElementById('checknewgrants')
    btn.classList.add('disabled')
    btn.disabled = true

    addLoadingCircle('maincontainer')
    addAwdPropTables('maincontainer')

    $.ajax( {
        url: '/findnewgrants',
        type: 'POST',
        data: '',
        processData: false,
        contentType: false,
        success: function(result) {
            document.getElementById('loading-circle').remove()

            if (result.includes('Searching finished...')) {
                console.log('true returned')
            }
        }
    });

    e.preventDefault();
});

/**
 * UI element creation
 * 
 * Codes for creating buttons, loading circles, tables 
 */
function insertDoUpdateButton(element_id) {
    var btn = document.createElement('button');
    btn.id = "checknewgrants";
    btn.className = "btn btn-primary btn-lg btn-block mt-4";
    btn.type = "button";
    btn.innerHTML = "Check for new grants";

    document.getElementById(element_id).appendChild(btn);
}

function addLoadingCircle(element_id) {
    var div1 = document.createElement('div');
    div1.className = 'text-center';
    div1.id = 'loading-circle';

    var div2 = document.createElement('div');
    div2.className = 'spinner-border text-primary mt-2'
    div2.role = 'status';

    var span1 = document.createElement('span');
    span1.className = 'sr-only';

    div2.appendChild(span1);
    div1.appendChild(div2);

    document.getElementById(element_id).appendChild(div1)
}

function addAwdPropTables(element_id) {
    awdTable = `
        <h5 class='h5 text-center mt-3'>New Awards</h5>
        <div class="table-responsive-sm mt-2" style="overflow: auto; height: 244px; max-height: 244px;">
            <table id="awardTable" class="table table-striped table-bordered table-sm" >
                <thead style="position: sticky; top: 0; background: #eeeeee;">
                    <tr>
                    <th scope="col">Proposal #</th>
                    <th scope="col">Award Title</th>
                    <th scope="col">Award Amount</th>
                    <th scope="col">PI</th>
                    <th scope="col">Co-PIs</th>
                    <th scope="col">Date</th>
                    <th scope="col">College</th>
                    <th scope="col">Dept</th>
                    </tr>
                </thead>

                <tbody id="awardTableBody">
                    
                </tbody>
            </table>
        </div>`

    propTable = `
        <h5 class='h5 text-center mt-3'>New Proposals</h5>
        <div class="table-responsive-sm mt-2" style="overflow: auto; height: 244px; max-height: 244px;">
            <table id="proposalTable" class="table table-striped table-bordered table-sm" >
                <thead style="position: sticky; top: 0; background: #eeeeee;">
                <tr>
                    <th scope="col">Proposal #</th>
                    <th scope="col">Proposal Title</th>
                    <th scope="col">Total Funds</th>
                    <th scope="col">PI</th>
                    <th scope="col">Co-PIs</th>
                    <th scope="col">Date</th>
                    <th scope="col">College</th>
                    <th scope="col">Dept</th>
                    <th scope="col">Status</th>
                </tr>
                </thead>

                <tbody id="proposalTableBody">
                
                </tbody>
      </table>
    </div>`

    $("#"+element_id).append(awdTable);
    $("#"+element_id).append(propTable);
}

/**
 * Below are the functions for adding rows to the award and proposal tables
 * 
 */
function addAwdRow(data) {
    table = document.getElementById('awardTableBody');

    var row = table.insertRow(-1);
    var cell1 = row.insertCell(0);
    var cell2 = row.insertCell(1);
    var cell3 = row.insertCell(2);
    var cell4 = row.insertCell(3);
    var cell5 = row.insertCell(4);
    var cell6 = row.insertCell(5);
    var cell7 = row.insertCell(6);
    var cell8 = row.insertCell(7);

    cell1.innerHTML = data['Prop-num'];
    cell2.innerHTML = data['Award-title'];
    cell3.innerHTML = data['Award Amount'];
    cell4.innerHTML = data['PI'];
    cell5.innerHTML = data['Other-personnel'];
    cell6.innerHTML = data['Date'];
    cell7.innerHTML = data['College'];
    cell8.innerHTML = data['Dept'];
}

function addPropRow(data) {
    table = document.getElementById('proposalTableBody');

    var row = table.insertRow(-1);
    var cell1 = row.insertCell(0);
    var cell2 = row.insertCell(1);
    var cell3 = row.insertCell(2);
    var cell4 = row.insertCell(3);
    var cell5 = row.insertCell(4);
    var cell6 = row.insertCell(5);
    var cell7 = row.insertCell(6);
    var cell8 = row.insertCell(7);
    var cell9 = row.insertCell(8);

    cell1.innerHTML = data['Prop-num'];
    cell2.innerHTML = data['Award-title'];
    cell3.innerHTML = data['Total Funds'];
    cell4.innerHTML = data['PI'];
    cell5.innerHTML = data['Other-personnel'];
    cell6.innerHTML = data['Date'];
    cell7.innerHTML = data['College'];
    cell8.innerHTML = data['Dept'];
    cell9.innerHTML = data['Status'];
}
