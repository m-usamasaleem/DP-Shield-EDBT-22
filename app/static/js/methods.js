function dropdownChange() {
    var im_path = "../static/img/" + document.getElementById("inputDropdown").value + ".png";

    console.log("Changing to " + im_path);
    document.getElementById("inputImageDisplay").src = im_path;   
}


var radios = document.querySelectorAll('input[type=radio][name="methodSelection"]');

function changeHandler(event) {
    $("#btnGenPrivImage").prop("disabled", false);

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


$(document).on('click','#btnGenPrivImage', function(e) {
    var selectedMethod = document.querySelector('input[name="methodSelection"]:checked').value;
    var im_path = "../static/img/" + document.getElementById("inputDropdown").value + ".png";

    $( $("input[name=methodSelection], .custom-range, #inputDropdown") ).each(function( i, e ) {
        $(e).prop("disabled", true);
    });

    var toSend;

    if (selectedMethod == "dpsamp") {
        toSend = {"method": selectedMethod, "input_image_path": im_path, "epsilon": $("#dpsampEpsilonSlider")[0].value, 'cluster_sz': $("#dpsampClusterSzSlider")[0].value, 'mval': $("#dpsampNumPixSlider")[0].value};
    } else if (selectedMethod == "dppix") {
        toSend = {"method": selectedMethod, "input_image_path": im_path, "epsilon": $("#dppixEpsilonSlider")[0].value, 'block_sz': $("#dppixBlockSzSlider")[0].value, 'mval': $("#dppixNumPixSlider")[0].value};
    } else if (selectedMethod == "dpsvd") {
        toSend = {"method": selectedMethod, "input_image_path": im_path, "epsilon": $("#dpsvdEpsilonSlider")[0].value, 'eigen_sz': $("#dpsvdEigenSzSlider")[0].value};
    } else if (selectedMethod == "snow") {
        toSend = {"method": selectedMethod, "input_image_path": im_path, "delta": $("#snowDeltaSlider")[0].value};
    };

    $("#dpsampVisuals, #dppixVisuals, #dpsvdVisuals, #snowVisuals").addClass("hide");
    $("#privacyAttackVisuals").addClass("hide");

    $.ajax( {
        url: '/methods',
        type: 'POST',
        data: JSON.stringify(toSend),
        contentType: "application/json; charset=utf-8",
        success: function(result) {
            console.log("Success " + result);
            return_dict = JSON.parse(result.toString())
            console.log(return_dict)
            timestamp = return_dict["save_time"];

            // Visuals
            $("#dpsampInputIm")[0].src = im_path;
            $("#dppixInputIm")[0].src = im_path;
            $("#dpsvdInputIm")[0].src = im_path;
            $("#snowInputIm")[0].src = im_path;

            if (selectedMethod == "dpsamp") {
                $("#dpsampSampledIm")[0].src = "../static/uploads/samp_sampledim_" + timestamp + ".jpg";
                $("#dpsampPrivateImage")[0].src = "../static/uploads/samp_private_" + timestamp + ".jpg";
            } else if (selectedMethod == "dppix") {
                $("#dppixNoDP")[0].src = "../static/uploads/pix_blur_nodp_" + timestamp + ".jpg";
                $("#dppixPrivateImage")[0].src = "../static/uploads/pix_blur_dp_" + timestamp + ".jpg";
            } else if (selectedMethod == "dpsvd") {
                $("#dpsvdNoDP")[0].src = "../static/uploads/svd_nopriv_" + timestamp + ".jpg";
                $("#dpsvdPrivateImage")[0].src = "../static/uploads/svd_priv_im_" + timestamp + ".jpg";
            } else if (selectedMethod == "snow") {
                $("#snowPrivateImage")[0].src = "../static/uploads/snow_priv_" + timestamp + ".jpg";
            };

            mse = (Math.round(return_dict['privMSE']*10000)/10000).toString();
            $("#" + selectedMethod + "MSE")[0].innerText = "MSE: " + mse;

            ssim = (Math.round(return_dict['privSSIM']*10000)/10000).toString();
            $("#" + selectedMethod + "SSIM")[0].innerText = "SSIM: " + ssim;

            $("#"+selectedMethod+"Visuals").removeClass("hide");



            // Privacy attacks
            $("#realIdText")[0].innerText = "True ID: " + return_dict['groundtruth_id'];

            ids = return_dict['top_5_ids'];
            probs = return_dict['top_5_probabilities'];
            id_ims = return_dict['top_5_impaths'];
            for (let i = 0; i < 5; i++) {
                id = ids[i];
                prob = probs[i];
                prob = (Math.round(prob*1000)/10).toString()+"%";
                impath = id_ims[i];
                imStr = "tableIm"+(i+1);
                
                // idStr = "tableID"+(i+1);
                // probStr = "tableProb"+(i+1);
                tableText = "tableText"+(i+1);

                $("#"+tableText)[0].innerText = "ID: " + id + " Probability="+prob;
                $("#"+imStr)[0].src = "../static/" + impath;
                // $("#"+idStr)[0].innerHTML = id;
                // $("#"+probStr)[0].innerHTML = (Math.round(prob*1000)/10).toString()+"%";
            }

            ids = return_dict['top_5_ids_input'];
            probs = return_dict['top_5_probabilities_input'];
            id_ims = return_dict['top_5_impaths_input']
            for (let i = 0; i < 5; i++) {
                id = ids[i];
                prob = probs[i];
                prob = (Math.round(prob*1000)/10).toString()+"%";
                impath = id_ims[i];
                imStr = "inpTableIm"+(i+1);
                
                // idStr = "tableID"+(i+1);
                // probStr = "tableProb"+(i+1);
                tableText = "inpTableText"+(i+1);

                $("#"+tableText)[0].innerText = "ID: " + id + " Probability="+prob;
                $("#"+imStr)[0].src = "../static/" + impath;
                // $("#"+idStr)[0].innerHTML = id;
                // $("#"+probStr)[0].innerHTML = (Math.round(prob*1000)/10).toString()+"%";
            }

            $("#privacyAttackVisuals").removeClass("hide");
        },
        error: function(result) {
            console.log("Error " + result);
        }
    });

    $( $("input[name=methodSelection], .custom-range, #inputDropdown") ).each(function( i, e ) {
        $(e).prop("disabled", false);
    });
});