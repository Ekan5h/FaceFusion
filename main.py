import cv2
import numpy as np
import base64

triangles = [
    [0,1,4],[1,4,5],[1,3,5],[5,3,6],
    [3,2,6],[2,7,6],[0,4,7],[0,2,7],
    [4,5,11],[5,11,12],[5,6,12],[6,9,12],
    [6,7,9],[7,9,10],[4,10,11],[4,7,10],
    [8,11,12],[8,9,12],[8,9,10],[8,10,11]
]
adj = [[] for _ in range(20)]

for i in range(20):
    for j in range(20):
        if i==j: continue
        if len([v for v in triangles[i] if v in triangles[j]])==2:
            adj[i].append(j)
for i in range(20):
    for j in range(20):
        if i==j: continue
        if len([v for v in triangles[i] if v in triangles[j]])==1:
            adj[i].append(j)

adj[2].append(7)
adj[4].append(0)

I = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])


def findTriangle(point, t, P, adj, triangles):
    if inTriangle(point, P, triangles[t]):
            return t
    for i in adj[t]:
        if inTriangle(point, P, triangles[i]):
            return i
    for i in range(len(triangles)):
        if inTriangle(point, P, triangles[i]):
            return i

def inTriangle(point,P, tri):
    A = area(P[tri[0]], P[tri[1]], P[tri[2]])
    A1 = area(point, P[tri[1]], P[tri[2]])
    A2 = area(P[tri[0]], point, P[tri[2]])
    A3 = area(P[tri[0]], P[tri[1]], point)
    if abs(A-(A1+A2+A3))<1e-9:
        return True
    return False
    

def area(point1, point2, point3): 
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x3 = point3[0]
    y3 = point3[1]
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  + x3 * (y1 - y2)) / 2.0) 

def preprocess(fname, model):
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    
    scale = 1

    if(img.shape[0]<=img.shape[1]):
        imgr = cv2.resize(img,(256, (img.shape[0]*256)//img.shape[1]))
        scale = img.shape[1]/256
    else:
        imgr = cv2.resize(img,((img.shape[1]*256)//img.shape[0], 256))
        scale = img.shape[0]/256

    faces = model.detect_faces(imgr)

    if not len(faces):
        raise Exception("No face found!")
    faces.sort(key=lambda x: x['confidence'], reverse=True)


    face = faces[0]
    bBox = face['box']
    for i in range(4): bBox[i] = int(scale*bBox[i])

    try:
        img = img[
            max(int(bBox[1]-bBox[3]*0.2),0):min(int(bBox[1]+1.2*bBox[3]),img.shape[1]),
            max(int(bBox[0]-bBox[2]*0.2),0):min(int(bBox[0]+1.2*bBox[2]),img.shape[0])
            ]
    except:
        raise Exception("Face too close to boundary!")

    # cv2.imshow("face", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    t2 = lambda x: [scale*x[0]-max((bBox[0]-bBox[2]*0.2),0), scale*x[1]-max((bBox[1]-bBox[3]*0.2),0)]
    scalex = 192/img.shape[1]
    scaley = 256/img.shape[0]
    t1 = lambda x: [x[0]*scalex, x[1]*scaley]
    t = lambda x: t1(t2(x))

    img = cv2.resize(img, (192, 256))

    tiepoints = np.array([
        [0,0],
        [191,0],
        [0,255],
        [191,255],
        t1((0.2*bBox[2], 0.2*bBox[3])),
        t1((1.2*bBox[2], 0.2*bBox[3])),
        t1((1.2*bBox[2], 1.2*bBox[3])),
        t1((0.2*bBox[2], 1.2*bBox[3])),
        t(face['keypoints']['nose']),
        t(face['keypoints']['mouth_right']),
        t(face['keypoints']['mouth_left']),
        t(face['keypoints']['left_eye']),
        t(face['keypoints']['right_eye'])
    ])

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    jpg_as_text = base64.b64encode(buffer)
    # print(jpg_as_text)

    return {
        'img':str(jpg_as_text),
        'tie_points':tiepoints.tolist()
    }

def morph(fname1, fname2, l, P1, P2):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    
    # P1 = img1['tie_points']
    # P2 = img2['tie_points']
    # img1 = img1['img']
    # img2 = img2['img']

    # output = np.zeros((200,150,3), np.uint8)
    output = [[[] for _ in range(192)] for _ in range(256)]

    transforms = []

    for tri in triangles:
        M1 = np.array([
            [P1[tri[0]][0], P1[tri[1]][0], P1[tri[2]][0]],
            [P1[tri[0]][1], P1[tri[1]][1], P1[tri[2]][1]],
            [1, 1, 1]
        ])
        M2 = np.array([
            [P2[tri[0]][0], P2[tri[1]][0], P2[tri[2]][0]],
            [P2[tri[0]][1], P2[tri[1]][1], P2[tri[2]][1]],
            [1, 1, 1]
        ])

        transform = M2@np.linalg.inv(M1)
        transform1 = np.linalg.inv((1-l)*I+l*transform)
        transform2 = transform1@transform
        transforms.append([transform1, transform2])

    P = (1-l)*P1+l*P2
    print("Painting!")
    t = 0 
    for i in range(256):
        for j in range(192):
            t = findTriangle([j,i], t, P, adj, triangles)
            point1 = [
                round(j*transforms[t][0][0][0] + i*transforms[t][0][0][1] + transforms[t][0][0][2]),
                round(j*transforms[t][0][1][0] + i*transforms[t][0][1][1] + transforms[t][0][1][2])
            ]
            point2 = [
                round(j*transforms[t][1][0][0] + i*transforms[t][1][0][1] + transforms[t][1][0][2]),
                round(j*transforms[t][1][1][0] + i*transforms[t][1][1][1] + transforms[t][1][1][2])
            ]
            try:
                output[i][j] = (1-l)*img1[point1[1]][point1[0]] + l*img2[point2[1]][point2[0]]
            except:
                pass
    print("Done!")
    output = np.array(output, np.uint8)
    cv2.putText(output, "FaceFusion", (10,246), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)
    # cv2.imshow("face", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    _, buffer = cv2.imencode('.jpg', output)
    jpg_as_text = base64.b64encode(buffer)
    return str(jpg_as_text)

# morph("/home/ekansh/Downloads/Images/greeen.jpeg", "/home/ekansh/Desktop/DevSpace/Face-Morphing/Sample/from.jpg",0.3)
# morph("/home/ekansh/Downloads/khyati1.jpeg", "/home/ekansh/Downloads/khyati.jpeg",0.5)
# morph("/home/ekansh/Downloads/Images/greeen.jpeg", "/home/ekansh/Downloads/khyati1.jpeg",0.7)
# morph("/home/ekansh/Downloads/khyati.jpeg", "obama.jpg",0.5)
# morph('../Face-Morphing/Sample/to.jpg', 'obama.jpg', 0.5)
