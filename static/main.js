face = [
    document.getElementById("facel1"),
    document.getElementById("facel2")
];

tiePoints = [0,1]

function changeFile(f,id){
    if(f.value){
        temp = f.value.split("\\");
    }else{
        temp = ["Select file..."]
    }
    face[id].innerText = temp[temp.length-1].substr(0,17);
    if(temp[temp.length-1].length>17){
        face[id].innerText = face[id].innerText + "...";
    }
    if(!f.value){
        return 0;
    }

    let a = new FormData();
    a.append("img",f.files[0]);
    let xhr = new XMLHttpRequest();
    xhr.open('post', "/api/preprocess/", true);
    document.getElementById("overlay").style.display = "block";
    xhr.onload = ()=>{
                    try{
                        resp = JSON.parse(xhr.response);
                        tiePoints[id] = resp.tie_points;
                        let url = resp.img.substr(2,resp.img.length-3);
                        document.getElementById("img"+(id+1)).src = "data:image/jpeg;base64,"+url;
                    }catch{
                        alert("Some error occurred! Make sure there is a face in the image.");
                    }
                    document.getElementById("overlay").style.display = "none";
                }
    xhr.send(a);
}


function morph(){
    img1 = document.getElementById("img1").src;
    img2 = document.getElementById("img2").src;
    l = document.getElementById("ratio").value;
    t1 = tiePoints[0];
    t2 = tiePoints[1];
    let a = new FormData()
    a.append('img1', img1);
    a.append('img2', img2);
    a.append('l', l);
    a.append('t1', JSON.stringify(t1));
    a.append('t2', JSON.stringify(t2));
    let xhr = new XMLHttpRequest();
    xhr.open('post', "/api/morph/", true);
    document.getElementById("overlay").style.display = "block";
    xhr.onload = ()=>{
                    try{
                        resp = JSON.parse(xhr.response);
                        let url = resp.img.substr(2,resp.img.length-3);
                        document.getElementById("result").src = "data:image/jpeg;base64,"+url;
                        document.getElementById("result1").src = "data:image/jpeg;base64,"+url;
                    }catch{
                        alert("Some error occurred!");
                    }                    
                    document.getElementById("overlay").style.display = "none";
                }
    xhr.send(a);
}

window.onload = () => {
        document.getElementById("overlay").style.display = "none";
        document.getElementById("msg").style.display = "block";
    }
