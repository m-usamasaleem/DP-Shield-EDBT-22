function dropdownChange() {
    var dataset = document.getElementById("inputDropdown").value;
    $("#vggfaces2, #casiafaces").addClass("hide");

    if (dataset != "noneselected") {
        $("#"+dataset).removeClass("hide");   
    }
}