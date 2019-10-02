
function api_post_generate(files, type) {

    var file = files[0];
    var data = new FormData();
    data.append('model', file);
    data.append('model_type', type);

    spinner.style.display = 'block';

    var request = new XMLHttpRequest();
    request.addEventListener('load', function(e) {

        spinner.style.display = 'none';

        if (request.response['error']) {
            alert(request.response['error']);
        } else {
            build(request.response['result'], request.response['model_type']);
        }
    });

    request.upload.addEventListener('progress', function(e) {
        var percent_complete = (e.loaded / e.total) * 100;                
        console.log(percent_complete);
    });

    request.responseType = 'json';
    request.open('post', '/generate'); 
    request.send(data);

} 

function api_get_predefined(index, type) {

    spinner.style.display = 'block';

    var request = new XMLHttpRequest();

    request.addEventListener('load', function(e) {

        spinner.style.display = 'none';

        if (request.response['error']) {
            alert(request.response['error']);
        } else {
            build(request.response['result'], request.response['model_type']);
        }
    });

    request.upload.addEventListener('progress', function(e) {
        var percent_complete = (e.loaded / e.total) * 100;                
        console.log(percent_complete);
    });

    request.responseType = 'json';
    request.open('get', '/predefined/' + index + '/' + type); 
    request.send();

} 