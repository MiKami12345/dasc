function submitExam() {
    var form = $('#drivingExamForm')[0];
    var formData = new FormData(form);

    $.ajax({
        type: "POST",
        url: "/questions",
        data: formData,
        contentType: false,
        processData: false,
        success: function(response) {
            $('#result').html(response);
        }
    });
    console.log("Submit button clicked!");
    // フォームの実際のsubmitを発生させる
    form.submit();
}
