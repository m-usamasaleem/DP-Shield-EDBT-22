var radios = document.querySelectorAll('input[type=radio][name="methodSelection"]');

function changeHandler(event) {
    $("#btnDoClustering").prop("disabled", false);

    document.getElementById("dpsampDiv").classList.add("hide");
    document.getElementById("dppixDiv").classList.add("hide");
    document.getElementById("dpsvdDiv").classList.add("hide");
    document.getElementById("snowDiv").classList.add("hide");

    var targetID = this.value + "Div";
    document.getElementById(targetID).classList.remove("hide");
}

Array.prototype.forEach.call(radios, function(radio) {
   radio.addEventListener('change', changeHandler);
});


$(document).on('click','#btnDoClustering', function(e) {
    var selectedDataset = document.querySelector('input[name="datasetSelection"]:checked').value;
    var selectedMethod = document.querySelector('input[name="methodSelection"]:checked').value;

    console.log(selectedDataset);
    console.log(selectedMethod);

    var toSend;

    if (selectedMethod == "dpsamp") {
        toSend = {"method": selectedMethod, "dataset_name": selectedDataset, "epsilon": $("#dpsampEpsilonSlider")[0].value, 'cluster_sz': $("#dpsampClusterSzSlider")[0].value, 'mval': $("#dpsampNumPixSlider")[0].value};
    } else if (selectedMethod == "dppix") {
        toSend = {"method": selectedMethod, "dataset_name": selectedDataset, "epsilon": $("#dppixEpsilonSlider")[0].value, 'block_sz': $("#dppixBlockSzSlider")[0].value, 'mval': $("#dppixNumPixSlider")[0].value};
    } else if (selectedMethod == "dpsvd") {
        toSend = {"method": selectedMethod, "dataset_name": selectedDataset, "epsilon": $("#dpsvdEpsilonSlider")[0].value, 'eigen_sz': $("#dpsvdEigenSzSlider")[0].value};
    } else if (selectedMethod == "snow") {
        toSend = {"method": selectedMethod, "dataset_name": selectedDataset, "delta": $("#snowDeltaSlider")[0].value};
    };

    console.log(toSend);
    $("#clusteringLoading").removeClass("hide");
    $("#visImageDisplay").addClass("hide");

    // $("#dpsampVisuals, #dppixVisuals, #dpsvdVisuals, #snowVisuals").addClass("hide");
    // $("#privacyAttackVisuals").addClass("hide");

    $.ajax( {
        url: '/clustering',
        type: 'POST',
        data: JSON.stringify(toSend),
        contentType: "application/json; charset=utf-8",
        success: function(result) {
            console.log("Success " + result);
            timestamp = result;

            $("#visImage")[0].src = "../static/uploads/" + timestamp + "vis2.png";

            $("#visImageDisplay").removeClass("hide");
            $("#clusteringLoading").addClass("hide");
        },
        error: function(result) {
            console.log("Error " + result);
            $("#clusteringLoading").addClass("hide");
        }
    });



    // $( $("input[name=methodSelection], .custom-range, #inputDropdown") ).each(function( i, e ) {
    //     $(e).prop("disabled", false);
    // });
});