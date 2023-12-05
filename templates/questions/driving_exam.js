function submitExam() {
    var form = $('#drivingExamForm')[0];
    var formData = new FormData(form);

    $.ajax({
        type: "POST",
        url: "/questions/check_exam",
        data: formData,
        contentType: false,
        processData: false,
        success: function(response) {
            $('#result').html(response);
        }
    });
}
