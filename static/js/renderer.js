var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 10000);

var renderer = new THREE.WebGLRenderer({antialias: true});
renderer.shadowMap.enabled = true;
renderer.shadowMapType = THREE.PCFSoftShadowMap;
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

var spotLight = new THREE.SpotLight( 0xFFFFFF, 0.7);
spotLight.position.set(2, 2, 50);
spotLight.target.position.set(0, 0, 0);
spotLight.castShadow = true;
scene.add(spotLight.target);
scene.add(spotLight);
spotLight.shadow.mapSize.width = 512; // default
spotLight.shadow.mapSize.height = 512; // default
spotLight.shadow.camera.near = 0.5; // default
spotLight.shadow.camera.far = 15000; // default


var ambientLight = new THREE.AmbientLight( 0x888888 );
scene.add(ambientLight);

renderer.setClearColor( 0xffffff, 1 );

var controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.set(24, 0, 12);
camera.lookAt(0, 0, 0);
camera.up.set( 0, 0, 1 );
controls.update();

var geometry = new THREE.PlaneGeometry(60, 60, 1, 1);
var material = new THREE.MeshLambertMaterial( {color: 0xffffff} );
var plane = new THREE.Mesh(geometry, material);
plane.position.set(0, 0, 0);
plane.receiveShadow = true;
scene.add(plane);

// var axes = new THREE.AxisHelper(50);
// scene.add(axes);

var loader = new THREE.OBJLoader();

var block1 = null;
var block2 = null;
var block3 = null;

var object = null;

[1, 2, 3].forEach((val, idx) => {

    loader.load(
        '/static/model/' + val + '.obj',
        function (object) {

            var obj = object.children[0];
            
            if (val == 1) {
                block1 = obj;
            } else if (val == 2) {
                block2 = obj;
            } else if (val == 3) {
                block3 = obj;
            }

            obj.castShadow = true;
            obj.receiveShadow = true;
    
        },
        function (xhr) {
    
            console.log(val + ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
    
        },
        function (error) {
    
            console.log( 'An error happened for ' + val);
    
        }
    );

});

var animate = function () {
    requestAnimationFrame(animate);
    
    if (object) {
        object.rotation.z += 0.006;
    }

    controls.update();
    renderer.render(scene, camera);
};

animate();

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

function generate_lego(dic) {
    
    var total = Object.keys(dic).length - 2;

    var configurations = [];
    
    for (var i = 0; i < total; i++) {

        var positions = dic[i];
        var n = positions.length;
        var rotation = 0;

        if (n == 1) {
        
            rotation = 0
        
        } else {

            var x1 = parseInt(positions[0]['x']);
            var y1 = parseInt(positions[0]['y']);
            var x2 = parseInt(positions[1]['x']);
            var y2 = parseInt(positions[1]['y']);
            
            if (x2 > x1) {
                rotation = 0;
            } else if (y2 > y1) {
                rotation = 90;
            } else if (x2 < x1) {
                rotation = 180;
            } else if (y2 < y1) {
                rotation = 270;
            }

        }

        var configuration = {
            
            'num': n,
            'rotation': rotation,
            'x': parseInt(positions[0]['x']),
            'y': parseInt(positions[0]['y']),
            'z': parseInt(positions[0]['z'])
        }

        configurations.push(configuration)

    }

    var legos = [];

    var prev_h = -1;

    var x_sum = 0;
    var y_sum = 0;
    var z_sum = 0;

    for (var i = 0; i < configurations.length; i++) {
        
        var c = configurations[i];

        var rotation = c['rotation'];
        var num = c['num'];
        var x   = c['x'];
        var y   = c['y'];
        var z   = c['z'];

        var mesh;

        if (num == 1) {
            mesh = block1.clone();
        } else if (num == 2) {
            mesh = block2.clone();
        } else if (num == 3) {
            mesh = block3.clone();
        }

        if (rotation != 0) {
            var radian = rotation * Math.PI / 180;
            mesh.rotateZ(radian);  
        }

        var h, s, l;
        
        while (true) {

            var candidates = [0, 9, 18, 35, 64, 85];
            var r = getRandomInt(candidates.length);
            var h = candidates[r];
            
            if (h != prev_h) {
                prev_h = h;
                break
            }
        }
    
        h = (h * 360 / 100) | 0;
        l = 50;
        s = 100;

        var color = new THREE.Color('hsl(' + h + ', ' + s + '%, ' + l + '%)');

        var material = new THREE.MeshPhongMaterial( { color: color, reflectivity: 1.0 } );

        mesh.material = material;

        mesh.position.x = x + 0.5;
        mesh.position.y = y + 0.5;
        mesh.position.z = z + 0.5;
        
        legos.push(mesh);

        x_sum += mesh.position.x;
        y_sum += mesh.position.y;
        z_sum += mesh.position.z;

    }

    var average_x = (x_sum / legos.length) | 0;
    var average_y = (y_sum / legos.length) | 0;
    var average_z = (z_sum / legos.length) | 0;

    plane.position.z = -average_z;

    var index    = 0; 
    function add() {

        if (index >= legos.length - 1) {
            clearInterval(interval);
            return;
        }

        var lego = legos[index];
        lego.position.x -= average_x;
        lego.position.y -= average_y;
        lego.position.z -= average_z;
        object.add(lego);

        index += 1;
    }
    var interval = setInterval(add, 40);

    scene.add(object);
}

function generate_pipe(dic) {

    var total = Object.keys(dic).length - 2;
    
    var vectors = [];

    var sum_x = 0;
    var sum_y = 0;
    var sum_z = 0;

    for (var i = 0; i < total; i++) {

        var positions = dic[i];

        var x = parseInt(positions[0]['x']);
        var y = parseInt(positions[0]['y']);
        var z = parseInt(positions[0]['z']);

        var vec = new THREE.Vector3(x, y, z);

        vectors.push(vec);

        sum_x += x;
        sum_y += y;
        sum_z += z;
    }

    var average_x = (sum_x / vectors.length) | 0;
    var average_y = (sum_y / vectors.length) | 0;
    var average_z = (sum_z / vectors.length) | 0;

    for (var i = 0; i < vectors.length; i++) {

        var vec = vectors[i];
        vec.x -= average_x;
        vec.y -= average_y;
        vec.z -= average_z;
    }

    var path = new THREE.CatmullRomCurve3(vectors, false, 'catmullrom', 10);

    var geometry = new THREE.TubeGeometry(path, 30, 0.2, 30, false);
    var material = new THREE.MeshPhongMaterial( { color: 0x666666, emissive: 0x000000 } );
    var mesh = new THREE.Mesh(geometry, material);

    mesh.castShadow = true;
    mesh.receiveShadow = true;

    plane.position.z = -average_z - 1;

    object.add(mesh);
    scene.add(object);
}


function build(dic, model_type) {

    if (object != null) {
        for (var i = 0; i < object.children.length; i++) {
            var child = object.children[i];
            scene.remove(child);
            child.geometry.dispose();
            child.material.dispose();
        }
        scene.remove(object);
        object = undefined;
    }
    
    object = new THREE.Group();

    if (model_type == 'lego') {
        generate_lego(dic);
    } else if (model_type == 'voxel') {
        generate_pipe(dic);
    } else {
        alert('The specified type of ' + model_type + ' is not available.')
    }

    camera.position.set(24, 0, 12);
    camera.lookAt(0, 0, 0);
    camera.up.set( 0, 0, 1 );

}

function download_model() {

    if (object == null) { return; }

    var exporter = new THREE.OBJExporter();
    var data = exporter.parse(object);
    var filename = 'model.obj';
    
    var blob = new Blob([data], {type: 'text/csv'});
    if(window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, filename);
    }
    else{
        var elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(blob);
        elem.download = filename;        
        document.body.appendChild(elem);
        elem.click();        
        document.body.removeChild(elem);
    }
}