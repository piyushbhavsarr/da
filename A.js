$(document).ready(function() {
  $("#name").keyup(function() {
    var name = $(this).val();
    if (name != "") {
      $.get("ajax.php", {name: name}, function(response) {
        $("#response").text(response);
      });
    } else {
      $("#response").text("Stranger, please tell me your name.");
    }
  });
});
