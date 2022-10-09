from copy import deepcopy
from math import ceil, floor, atan
import random
import numpy as np
from numpy import pi, cos, sin
import pygame
from sklearn.utils.extmath import cartesian
from PIL import Image

FPS = 30

"""
            GOLDEN RULE
    Pixel Level calls only in entire program!!

    TODO till break point one switch to neuro nets
      
    Remove angular and cartisian velocity and replace with a function. 
        Velocity as a function of time.
        If something bounces then the new function simply has a new orgin offset, and time offset to 0
    add to git hub for backup 
        deseret engine 
    --------------------------

    Neuro notes 
    Map complexity of the neruo net with Dr. As we add a layer, how does the amount of paths exponentiate
    Use noise sound and noise visual to show neuro net then high entropy as harmonic waves with amps(volumn as strength), and frequency as learning rate

    THought: Linear algebra allows you to transform feature space to solution space    
    
    
        Optimaization fork
    do a small project to understand how to profit from gpu linear algebra calls 
    learn to use a profiler to map where optimization is needed 
    Note that you have to learn how to code out the GPU for fast pixel level access
    refactor project so that triangles are multi threaded 
    
    Refactor clean up so that all transforms are in one step or at least add the 4th dimension so its cleaner

    Add ai that can identify things it sees in the vision feild 

    Add sounds of cube rotating in the wind. with doperler and postional sound. math of rotation 

    warp space time with gravity and so time slows orbits at points or something. also shape should morph as well as lights path

    Upon resume    
    Add light
    add shader
    further away more gray atmoshpic

    Upon resume
    Import complex poly sculpted in blender and render
    change textures to materials 
        rotate textures so they are none repeating
            randomize edges with wave so that overlap is obfuscated 

    refactor with quanternions
        refactor so vectors are 4d 
    
    Render in vr 

    Render in ar
"""
    
class Screen:
    #   Singelton Class 
    SCREEN_DIMENSIONS = np.array([1000, 1000], dtype=int)
    ASPECT_RATIO = SCREEN_DIMENSIONS[0] / SCREEN_DIMENSIONS[1]
    SCREEN_ORIGIN = np.array([0, 0], dtype=int) # Top left 
    #   Identity is the cordinates of each pixel at each of its cordinates. Used to transforms. Cartisian product used for notes
    SCREEN_IDENTITY = cartesian((np.arange(SCREEN_DIMENSIONS[0]), np.arange(SCREEN_DIMENSIONS[1]))).reshape(SCREEN_DIMENSIONS[0], SCREEN_DIMENSIONS[1], 2)
    SCREEN_IDENTITY = np.insert(SCREEN_IDENTITY, 2, 1, axis=2) #    We insert 1's as the third dimensions for transforms
    display = pygame.display.set_mode(SCREEN_DIMENSIONS)
    WHITE_SCREEN = np.ones(shape=(SCREEN_DIMENSIONS[0], SCREEN_DIMENSIONS[1], 3), dtype=np.int) * 255 # white screen

    def __init__(self) -> None:
        pygame.init()

    @staticmethod
    def update(pixel_array_rgb):
        """
            Argument is final pixel values. No background fill is used. All that proccessing should already be done at this point
            alpha means translucency metric
            No alpha channel here! Alpha never should interface with pygame. Alpha is managed in the rasterizer
            The weighted average of transluencent pixels and the first opaque pixel where alpha is wieght.  
        """

        #   Translate numpy 3d pixel array into a pygame surface object representing it.  
        surf = pygame.surfarray.make_surface(pixel_array_rgb)
        #   Paint the pixels to the buffer 
        Screen.display.blit(surf, Screen.SCREEN_ORIGIN) 

        #   Buffer to active display. Lock on screen resolved by deconstructor when screen falls out of scope  
        pygame.display.update()

class Transforms:
    
    @staticmethod
    def to_barycentric_cartesian(triangle):
        """
            All positions on a plane can be represented as a linear combination of the verticies of a triangle on that plane. 
            The sum of the coefficents of the linear combination for any point on the plane always equals 1. 
            If all coefficents are positive than the point is inside the triangle 
            We see three equations and three unkowns 
            See bay_to_original for coefficient matrix representing the 3 equations. Note bottom equation is all barycentric cordinates = 1

        """
        #bay_to_cartesian = np.vstack([triangle, [1,1,1]])
        #return np.linalg.inv(bay_to_cartesian) 
        return np.linalg.inv(triangle) 

    @staticmethod
    def origin_center_to_topleft():
        TRANSLATION_CENTER_TO_TOPLEFT_ORIGIN = np.array([\
        [1, 0, 0, int(Screen.SCREEN_DIMENSIONS[0] / 2)],
        [0, 1, 0, int(Screen.SCREEN_DIMENSIONS[1] / 2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

        
        # return np.array([Screen.SCREEN_DIMENSIONS / 2], dtype=int).transpose() #  Translate: local space to pixel space
        return TRANSLATION_CENTER_TO_TOPLEFT_ORIGIN

    @staticmethod
    def euler_to_quaternion(heading, pitch, bank):
        param = [heading/2, pitch/2, bank/2]
        ch, cp, cb = np.cos(param)
        sh, sp, sb = np.sin(param)
        return np.array([\
                ch*cp*cb + sh*sp*sb, 
                ch*sp*cb + sh*cp*sb, 
                sb*cp*cb - ch*sp*sb,
                ch*cp*sb - sh*sp*cb,
            ])
    
    @staticmethod  
    def quaternion_to_global_basis(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.array([\
            [1-2*y**2-2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
            [2*x*y - 2*w*z, 1-2*x**2-2*z**2, 2*y*z + 2*w*x],
            [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1-2*x**2-2*y**2] 
            ]) 

    @staticmethod
    def euler_to_global_basis(heading, pitch, bank):
        return Transforms.quaternion_to_global_basis(Transforms.euler_to_quaternion(heading, pitch, bank)) 

    @staticmethod
    def translate_and_rotate(heading, pitch, bank, position_3d, use_quaternions=False):
        t_r = np.zeros(shape=(4,4))
        if use_quaternions:
            t_r[:3,:3] = Transforms.euler_to_global_basis(heading, pitch, bank)
        else:
            t_r[:3,:3] = Transforms.rotation_matrix(heading, pitch, bank)
        t_r[:3,3] = position_3d
        t_r[3,3] = 1 
        return t_r

    @staticmethod
    def rotation_matrix(α, β, γ, translation=False):
        """
            Rotations are sequential different order different result. 
            Around the x axis by α then y by β, then z by γ. Differnet sequence different result
            Below is wrong. I never figured it out. Rotation is relative to the axis. So clockwise can be different signs for different axis
            Rotations are in radians, with positive as counter clockwise for x and y and then positive as clockwise for z. 
            Cordinate system is done with the left hand. thumb as y, up as positive. X as middle finger to the right as positive, pointer as z pointing away from you as positive
        """
        sα, cα = sin(α), cos(α)
        sβ, cβ = sin(β), cos(β)
        sγ, cγ = sin(γ), cos(γ)
        if translation is False:
            return np.array((
                (cβ*cγ,              -cβ*sγ,             sβ),
                (cα*sγ + sα*sβ*cγ,   cα*cγ - sγ*sα*sβ,   -cβ*sα),
                (sγ*sα - cα*sβ*cγ,   cα*sγ*sβ + sα*cγ,   cα*cβ)
            ))
        else:
            return np.array((
                (cβ*cγ,              -cβ*sγ,             sβ,     translation[0]),
                (cα*sγ + sα*sβ*cγ,   cα*cγ - sγ*sα*sβ,   -cβ*sα, translation[1]),
                (sγ*sα - cα*sβ*cγ,   cα*sγ*sβ + sα*cγ,   cα*cβ,  translation[2]) 
            ))
    
    @staticmethod
    def perspective():
        #   I use a homogenous vector. Meaning the 4th value in the vector is the demoninator for the first three elements
        #   https://www.youtube.com/watch?v=U0_ONQQ5ZNM&t=317s
        transform = np.zeros(shape=(4,4))
        transform[[0,1],[0,1]] = Camera.zoom
        #transform[2,[2,3]] = Camera.frustrum_back_clip_z + Camera.zoom, Camera.frustrum_back_clip_z*Camera.zoom   # this way requires massive numbers
        transform[2,2] = 1
        transform[3,2] = 1
        return transform

    @staticmethod
    def from_perspective_to_pixels():
        transform = np.zeros(shape=(4,4))
        #   Division part turns the perspective cordinates into range [-1, 1], numerator turns that into pixel cordinates with center of screen as origin
        transform[[0,1],[0,1]] = (Screen.SCREEN_DIMENSIONS/2) / Camera.front_clip_dimensions_halfed
        #   Invert y so that up is up and down is down. Note pygame has down as positive and origin as top left
        transform[1,1] *= -1
        #   Translate the origin so that its in the top left corner in pygame and logically still in the center
        #transform[[0,1], [3,3]] = Screen.SCREEN_DIMENSIONS/2
        #   Perserve z prime for rasterizer, and perserve homogenous vector z as denominator
        transform[[2,3],[2,3]] = 1
        return transform 

class Polygon:
    SMOTHEN = 1 # lessen jerky motion derived from random frame time

    """
        Verticies are in local cordinates with the center of gravity as the origin. 
        Else rotation occures in a unreal way. Also the nested origin simplfies the system in general
    """

    def __init__(self, name, verticies, triangle_indicies, triangle_norms, uv_map, scale=1, resolution=1, random_tilation=False):
        self.name = name
        self.scale = scale
        self.verticies = verticies
        self.triangle_indicies = triangle_indicies # Shape (n, 4). [(0-2 Triangle verticies indicies, 3 [surface normal,y,z]) ... ]   
        self.triangle_norms = triangle_norms
        self.velocity = np.array([0,0,0], dtype=float) # Meters/sec
        self.spin     = np.array((0,0,0), dtype=float) # Randians/sec
        self.orientation = np.array([0,0,0], dtype=float) # Eucldian angles in radians  
        self.center_of_gravity = np.array((0,0,0), dtype=float) # position of nested origin relative to global space
        self.uv_map = uv_map
        self.random_tilation = random_tilation

    def change_resolution_uniform(coefficient):
        """
            New Surface count = 3^coefficient * old Surface count
            We can get the center point of the triangle and then divide a triangle into 3 sub triangles on the same plane. 
            The object will not appear to have more resolution until the new verticies are warped with a transform. 
            The utility of this is that you can better see the effects of warping space without needing to manually define a high poly primitive. 
            For true higher definition of a primitive, construct the primitive with with a higher verticies count
        """
        pass

    def update(self, delta_time):
        delta_spin = self.spin * delta_time
        self.orientation = (self.orientation + delta_spin) % (2*pi) # division is to keep range between 0-2*pi
        self.center_of_gravity += delta_time * self.velocity
        return self.render()

    def render(self):
        #   Perform transforms but keep origin as center of gravity. Local transforms. Do not store the transforms
        rotated_verticies_and_translated = Transforms.rotation_matrix(*self.orientation, translation=self.center_of_gravity) @ self.verticies 
        rotated_triangle_norms = Transforms.rotation_matrix(*self.orientation) @ self.triangle_norms.transpose()
        rotated_scaled_and_translated = rotated_verticies_and_translated * self.scale
        
        return rotated_scaled_and_translated, rotated_triangle_norms

    def accelerate(self, velocity=0, spin=0):
        #   Dont use 
        self.spin += spin
        self.velocity += velocity 

    @staticmethod
    def get_normals(verticies, triangle_indicies):
        #   This method will only always work for uniform objects. If concavity is too high then the center of mass can be outside the object. 
        #   I time capped on finding a 3d way of boundary testing without knowing normals
        #   Idea is that the dot product the direction to the center of the polygon and the normal of the surface will tell if its facing out or in 
        #   https://stackoverflow.com/questions/57434507/determing-the-direction-of-face-normals-consistently
        triangle_normals = np.zeros_like(triangle_indicies)
        center_of_gravity = np.average(verticies, axis=1)
        for i, triangle in enumerate(triangle_indicies):
            triangle = verticies[:,triangle]
            triangle_center = np.average(triangle, axis=1)
            #   Unkown if facing inside or outside
            triangle_normal = np.cross(triangle[:,0] - triangle[:,1], triangle[:,0] - triangle[:,2])
            center_of_poly_to_triangle = triangle_center - center_of_gravity
            if np.dot(center_of_poly_to_triangle, triangle_normal) < 0:
                triangle_normal *= -1 # Normal was facing inside the polygon
            triangle_normals[i] = triangle_normal
        return triangle_normals

    @staticmethod
    def get_function_of_plane(point_on_plane, plane_normal):
        #   Find z given x and y points of plane 
        def find_z(x, y):
            return ((point_on_plane[0] - x)*plane_normal[0] + (point_on_plane[1] - y)*plane_normal[1])/plane_normal[2] + point_on_plane[2]
        return find_z

class Unit_cube(Polygon):
    #   Singelton. Transform and translate unit cube in sub class. This way memory stores only one cube and instances are expressed in terms of the same reference frame
    #   Todo: Throw a exception is constructor is called outside of sub class cube
    cube_count = 0
    verticies = np.array([\
            [-0.5, -0.5, -0.5, 1],
            [-0.5,  0.5, -0.5, 1],
            [ 0.5,  0.5, -0.5, 1],
            [ 0.5, -0.5, -0.5, 1],
            [-0.5, -0.5,  0.5, 1],
            [-0.5,  0.5,  0.5, 1],
            [ 0.5,  0.5,  0.5, 1],
            [ 0.5, -0.5,  0.5, 1]
        ]).transpose()
    triangle_indicies = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7], [0,5,4], [0,5,1], [2,3,6], [3,6,7], [0,3,4], [4,7,3], [1,2,5], [5,6,2]])
    triangle_normals = Polygon.get_normals(verticies[:3,:], triangle_indicies)
    #   Map the triangle faces with textures
    #uv_map = ["blue sand", "subca", "carpet", "mixed paint", "plie wood", "leaf", "drift wood", "gold sheets", "blue sand", "subca", "carpet", "mixed paint"]

    def __init__(self, scale=1, resolution=1, name="Cube ", random_tilation=False):
        Unit_cube.cube_count += 1 # update static var for class  +str(self.cube_count)
        uv_map = ["blue sand", "subca", "carpet", "mixed paint", "plie wood", "leaf", "drift wood", "gold sheets", "blue sand", "subca", "carpet", "mixed paint"]
        super().__init__(name=name+str(Unit_cube.cube_count), verticies=Unit_cube.verticies, triangle_indicies=Unit_cube.triangle_indicies,\
            triangle_norms=Unit_cube.triangle_normals, scale=scale, uv_map=uv_map, random_tilation=random_tilation)
        
class Camera:
    """
        You can change field of view, zoom and back clip at begining of loop, you just need to update the frustrum after with its method call
        TODO: zoom has no effect but appears working, add field of view buttons 
    """
    ORIENTATION_CARTESIAN_CAMERA_SPACE = np.array([0,0,1], dtype=float) #    Camera is always facing postive z in local space
    GLOBAL_TO_CAMERA_TRANSFORM = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float) # column 4 is position. Transform from this constant so no gimbal lock. 

    global_position = np.array([0,0,0], dtype=float)
    orientation_euler = np.array([0,0,0], dtype=float)
    global_to_camera_transform = np.copy(GLOBAL_TO_CAMERA_TRANSFORM) # column 4 is position
    zoom = 5
    field_of_view_y = pi * (3/4) # pi/2
    frustrum_back_clip_z = 10000 
    feild_of_view = np.array([Screen.ASPECT_RATIO*field_of_view_y, field_of_view_y])
    front_clip_dimensions_halfed = zoom * sin(feild_of_view) 
    local_to_persceptive_to_pixel_cordinates = None # Call update frustum before main loop so no cyclic dependency with Transform
    MOVEMENT_DELTA = 100 # Move x amount per second in global units 
    ROTATION_DELTA = (pi * 30 / 200) #   Rotate x radians per second
    ZOOM_DELTA = 100

    #   Rotating the orientation vector will give the normals of the frustrum
    IS_INSIDE_FRUSTRUM =    np.array([\
            np.append(Transforms.rotation_matrix(0,   pi/2 + feild_of_view[0]/2,    0) @ ORIENTATION_CARTESIAN_CAMERA_SPACE, 0),\
            np.append(Transforms.rotation_matrix(0,   -(pi/2 + feild_of_view[0]/2), 0) @ ORIENTATION_CARTESIAN_CAMERA_SPACE, 0),\
            np.append(Transforms.rotation_matrix((pi/2 + feild_of_view[0]/2), 0,  0) @ ORIENTATION_CARTESIAN_CAMERA_SPACE, 0),\
            np.append(Transforms.rotation_matrix(-(pi/2 + feild_of_view[0]/2), 0, 0) @ ORIENTATION_CARTESIAN_CAMERA_SPACE, 0),\
            np.array([0, 0, -1, zoom]),\
            np.array([0, 0, 1, -frustrum_back_clip_z]) 
        ], dtype=float) #    Constant to prevent gimbal lock. Transform frustrums normals in row 0-4 with euluer angles on update
    is_inside_frustrum = np.copy(IS_INSIDE_FRUSTRUM)
    
    @staticmethod
    def change_zoom(new_zoom):
        new_zoom = Camera.zoom + new_zoom*Camera.ZOOM_DELTA
        if new_zoom <= 1: 
            new_zoom = 1 # Min zoom
        Camera.frustrum_back_clip_z = new_zoom + 10000 # Change constant back clip offset here. 10000
        Camera.zoom = new_zoom
        #Camera.front_clip_dimensions_halfed = new_zoom * sin(Camera.feild_of_view)
        Camera.update_frustrum(Transforms.rotation_matrix(*[0,0,0])) 


    @staticmethod
    def update_frustrum(rotation_transform):
        Camera.feild_of_view = np.array([Screen.ASPECT_RATIO*Camera.field_of_view_y, Camera.field_of_view_y])
        Camera.front_clip_dimensions_halfed = Camera.zoom * sin(Camera.feild_of_view) 
        Camera.local_to_persceptive_to_pixel_cordinates = Transforms.from_perspective_to_pixels() @ Transforms.perspective()

        if Camera.zoom >= Camera.frustrum_back_clip_z:
            raise Exception("Front clip of frustram must be behine back")
        """   
            Rows of is inside frustrum:
                1 tests is point on right side of left clip of frustrum
                2 tests right 
                3 tests up 
                4 tests down
                5 tests front # Does not transform
                6 test back  # Does not transform
        """
        Camera.is_inside_frustrum[:4, :3] = (rotation_transform @ Camera.IS_INSIDE_FRUSTRUM[:4, :3].transpose()).transpose() 
        Camera.is_inside_frustrum[4,:] = np.array([0, 0, -1, Camera.zoom])
        Camera.is_inside_frustrum[5,:] = np.array([0, 0, 1, -Camera.frustrum_back_clip_z])

    @staticmethod
    def rotate_camera(delta_spin, delta_time=1):

        Camera.orientation_euler = (delta_spin*delta_time*Camera.ROTATION_DELTA + Camera.orientation_euler) % (2*pi) 
        #Camera.orientation_euler = (delta_spin + Camera.orientation_euler) % (2*pi) 


        #   I probably have the names wrong. Its rotation about x,y,z
        rotation_transform = Transforms.rotation_matrix(*Camera.orientation_euler)
        
        #   Rotate the basis vectors 
        Camera.global_to_camera_transform[:,:3] = rotation_transform @ Camera.GLOBAL_TO_CAMERA_TRANSFORM[:,:3]
    
        #   Rotate the frustrams normals from the new orientation cartesian
        Camera.update_frustrum(rotation_transform)


    @staticmethod
    def move(direction, delta_time=1):
        # move to new positon over time. Not teleportation
        #   TODO make direction a vectore that is scalled to slerp, then rotate it for movement
        Camera.global_position += direction * Camera.MOVEMENT_DELTA * delta_time
        Camera.global_to_camera_transform[:,3] -= direction #  Translation is opposite sign  

class Rasterize:

    @staticmethod
    def backface_culling(triangle_norms):
        #   If triangle is facing away from the camera then return False
        return (Camera.ORIENTATION_CARTESIAN_CAMERA_SPACE @ triangle_norms) < 0 # 0 means triangle face is ortho to the camera. ie a line and not a plane 

class Textures:
    """
        I take textures with arbitrary size. The large thier size the better thier resolution 
        I then compress the texture into a constant square size(Resolution) for all textures and store that in ram
        Each frame I resize the square into a rectange. Triangle mesh height and width as dimensions of rectangle
        I then use baysian transform to cut out overlapping pixels and paint the texture onto the trianglur face 
        
        Textures stored as rgba virtual square. Resolution, Resolution, 4

        Note the the face is constant. We are just seeing it at differnet resolutions. We never take a different size section of the texture
        We compress the image or tile it to make it bigger.  
    """

    RESOLUTION = 100
    textures = None  
    
    @staticmethod
    def load_virtual_textures():
        #   Load virtual textures into RAM
        Textures.textures = {\
            'blue sand':   np.array(Image.open('textures/virtual textures/blue sand.jpg')).swapaxes(1,0),\
            'carpet':      np.array(Image.open('textures/virtual textures/carpet.jpg')).swapaxes(1,0),\
            'mixed paint': np.array(Image.open('textures/virtual textures/mixed paint.jpg')).swapaxes(1,0),\
            'subca':       np.array(Image.open('textures/virtual textures/subca.jpg')).swapaxes(1,0),\
            'drift wood':  np.array(Image.open('textures/virtual textures/drift wood.jpg')).swapaxes(1,0),\
            'gold sheets': np.array(Image.open('textures/virtual textures/gold sheets.jpg')).swapaxes(1,0),\
            'leaf':        np.array(Image.open('textures/virtual textures/leaf.jpg')).swapaxes(1,0),\
            'plie wood':   np.array(Image.open('textures/virtual textures/plie wood.jpg')).swapaxes(1,0)\
                }

    @staticmethod
    def initialize_virtual_textures():
        #   Only call this once ever. Dont use on load. Use load
        #   Change the resolution of each texture independantly here by adding resolution x,y as second argument to reshape texture func. 
        #   Default resolution is 100, 100. Meaning that larger uses tile the texture to fill the face. 
        Textures.textures = {\
            'blue sand':   Textures.reshape_texture_to_rectange(np.array(Image.open('textures/blue sand.jpg')).swapaxes(1,0)),\
            'carpet':      Textures.reshape_texture_to_rectange(np.array(Image.open('textures/carpet.jpg')).swapaxes(1,0)),\
            'mixed paint': Textures.reshape_texture_to_rectange(np.array(Image.open('textures/mixed paint.jpg')).swapaxes(1,0)),\
            'subca':       Textures.reshape_texture_to_rectange(np.array(Image.open('textures/subca.jpg')).swapaxes(1,0)),\
            'drift wood':  Textures.reshape_texture_to_rectange(np.array(Image.open('textures/drift wood.jpg')).swapaxes(1,0)),\
            'gold sheets': Textures.reshape_texture_to_rectange(np.array(Image.open('textures/gold sheets.jpeg')).swapaxes(1,0)),\
            'leaf':        Textures.reshape_texture_to_rectange(np.array(Image.open('textures/leaf.jpg')).swapaxes(1,0)),\
            'plie wood':   Textures.reshape_texture_to_rectange(np.array(Image.open('textures/plie wood.jpg')).swapaxes(1,0))\
                }  
        
        #   Now appropriatly sized, we save to hard drive. Then on load we load these files and never pre process the original textures again 
        Image.fromarray(Textures.textures["blue sand"],    'RGB').save("textures/virtual textures/blue sand.jpg")
        Image.fromarray(Textures.textures["carpet"],       'RGB').save("textures/virtual textures/carpet.jpg")
        Image.fromarray(Textures.textures["mixed paint"],  'RGB').save("textures/virtual textures/mixed paint.jpg")
        Image.fromarray(Textures.textures["subca"],        'RGB').save("textures/virtual textures/subca.jpg")
        Image.fromarray(Textures.textures["drift wood"],   'RGB').save("textures/virtual textures/drift wood.jpg")
        Image.fromarray(Textures.textures["gold sheets"],  'RGB').save("textures/virtual textures/gold sheets.jpg")
        Image.fromarray(Textures.textures["leaf"],         'RGB').save("textures/virtual textures/leaf.jpg")
        Image.fromarray(Textures.textures["plie wood"],    'RGB').save("textures/virtual textures/plie wood.jpg")
          
    @staticmethod
    def reshape_texture_to_rectange(texture, rectange=(100, 100), random_tilation=False):
        """
            Hyper Optimize this function. It is called at fps*showing triangles / per sec
            NOTE: ! After some frustration I found that the array must have datatype np.uint8 inorder to work with the library
            Debug with
                img = Image.fromarray(texture, 'RGB')
                img.show()
        """

        t_width, t_height = texture.shape[0], texture.shape[1]
        w_step, h_step = t_width / rectange[0], t_height / rectange[1] # Defualt step of texture per pixel in rectangle

        #   Expand texture to fit rectange if rectange is too big. Do this by duplication of the tile and rotation to disrupt pattern 
        if w_step < 1 or h_step < 1: 
            #   Tile duplication to fit needed size. This way resolution is not lost. 
            x_tiles = ceil(rectange[0] / t_width)
            y_tiles = ceil(rectange[1] / t_height)
            reshaped_texture = np.tile(texture, (x_tiles, y_tiles,1)) 
            
            #   Pattern disruption algorithm. I crudely just rotate each row starting with a different rotation 
            for x_tile in range(x_tiles):
                for y_tile in range(y_tiles):
                    tile_slice = np.s_[x_tile*t_width: (1+x_tile)*t_width, y_tile*t_height: (1+y_tile)*t_height]
                    if random_tilation:
                        reshaped_texture[tile_slice] = np.rot90(texture, random.randrange(0,4)) # randomly rotate the texture tile so it looks less repeating
                    else:
                        reshaped_texture[tile_slice] = np.rot90(texture, x_tile + y_tile) # randomly rotate the texture tile so it looks less repeating

            #   Cutt off extra rows and columns
            reshaped_texture = reshaped_texture[:rectange[0], :rectange[1]]
            w_step, h_step = reshaped_texture.shape[0] / rectange[0], reshaped_texture.shape[1] / rectange[1] # Step of texture per pixel in rectangle
        
        else:
            reshaped_texture = np.zeros(shape=(rectange[0], rectange[1], 3), dtype=np.uint8)

        #   Compress the image to rectange size if texture is too big. Loss in resolution due to rectangle being far away  
        if w_step > 1 or h_step > 1:
            for i in range(rectange[0]):
                for ii in range(rectange[1]):
                    w_start, w_end = floor(w_step*i), floor(w_step*(i+1))
                    h_start, h_end = floor(h_step*ii), floor(h_step*(ii+1))
                    new_pixel = np.s_[w_start: w_end, h_start: h_end]
                    reshaped_texture[i, ii] = np.average(texture[new_pixel], axis=(0,1))
                    j = 2
        return reshaped_texture

class Controls:
    
    @staticmethod
    def update(delta_time):
        # TODO add zoom in out move back clip with zoom   
        """
            Move Camera
                w, s forward back

            Rotate Camera 

            Zoom 
                r,t zoom in out. ie move front clip of frustrum 
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                continue
            keys = pygame.key.get_pressed()

            #   Camera Movement
            camera_move = np.array([0,0,0])
            if keys[pygame.K_w]:
                camera_move[2] = 1
            if keys[pygame.K_s]:
                camera_move[2] = -1
            if keys[pygame.K_a]:
                camera_move[0] = 1
            if keys[pygame.K_d]:
                camera_move[0] = -1
            if keys[pygame.K_q]:
                camera_move[1] = 1
            if keys[pygame.K_e]:
                camera_move[1] = -1
            Camera.move(camera_move, delta_time)

            if keys[pygame.K_r]:
                Camera.change_zoom(1*delta_time) 
            if keys[pygame.K_t]:
                Camera.change_zoom(-1*delta_time)


            #   Camera Rotation
            camera_rotation = np.array([0,0,0])
            if keys[pygame.K_k]:
                camera_rotation[0] = 1
            if keys[pygame.K_i]:
                camera_rotation[0] = -1
            if keys[pygame.K_u]:
                camera_rotation[1] = 1
            if keys[pygame.K_o]:
                camera_rotation[1] = -1
            if keys[pygame.K_j]:
                camera_rotation[2] = 1
            if keys[pygame.K_l]:
                camera_rotation[2] = -1
            if np.any(camera_rotation != 0):
                Camera.rotate_camera(camera_rotation, delta_time)

#   Scene Object Creation
unit_cube =  Unit_cube(scale=400)
unit_cube_ = Unit_cube(scale=400)
unit_cube_.uv_map = ["gold sheets"] * 12

unit_cube.accelerate(spin=(pi/8, pi/4, pi/6)) 
unit_cube_.accelerate(velocity=(.1,0,0), spin=(0, -pi/4, -pi/6)) 

objects = [unit_cube, unit_cube_]

Screen() # Initiate Static Singleton
clock = pygame.time.Clock()
clock.tick()
Textures.load_virtual_textures() 
Camera.move(np.array([0,0,-20])) # Originate the camera so that the global origin is inside the frustrum 
Camera.rotate_camera(np.array([0, 0, 0])) #   Init frustrum

run = True
while run:
    
    delta_time = clock.tick(FPS) / 1000 # delta time is how many miliseconds its been since ticks last call. It itself is a slow function for some reason
    
    #   Process user input
    Controls.update(delta_time)
    
    #   Debug tools
    print(clock.get_fps())
    #print(np.round(Camera.orientation_euler / pi, 2)) 

    buffer_screen = deepcopy(Screen.WHITE_SCREEN) 
    # If a pixel is closer to the camera than the white background then overlap
    buffer_screen = np.insert(buffer_screen, 3, Camera.frustrum_back_clip_z, axis=2) #  TODO move this into screen no need to loop it, Change dtype to unit8 for pixels and int only for z with seperate array  

    for object in objects:
                
        cube_verticies, cube_triangle_norms = object.update(delta_time) # Objects's global cordinates
        
        #   Camera transform 
        global_to_camera = deepcopy(Camera.global_to_camera_transform)  
        #cube_triangle_norms = global_to_camera[:,:3] @ cube_triangle_norms #  Transform only. Reason. Normals dont have position, but verticies do
        global_to_camera[:3, :3] = global_to_camera.transpose()[:3,:3] #    Trivial transpose because verticies and norms are swapped shape 
        cube_verticies = global_to_camera[:3,:3] @ cube_verticies[:3,:] #    Transform and translate
        cube_verticies += (global_to_camera[:3,:3] @ -Camera.global_position).reshape(3,1)

        #   Temp speed dev TODO change size of transforms so this is not needed in refactor  
        cube_verticies = np.insert(cube_verticies, 3, 1, axis=0)

        #   Is inside Frustrum culling
        #   Check if verticies are inside the cameras frustrum 
        is_inside_frustrum = np.all((Camera.is_inside_frustrum @ cube_verticies) < 0, axis=0) #   Adding 4th row for operation
        triangles_inside_frustrum = np.full((12,1), False)
        #   Identify all triangles that dont have all three verticies inside the frustrum
        for i, triangle in enumerate(object.triangle_indicies):
            triangles_inside_frustrum[i] = np.all(is_inside_frustrum[triangle]) #  Make sure that all of the triangles verticies are inside the frustrum
        #   Cull out triangles who have any verticies outside the frustrum
        triangle_indicies   = object.triangle_indicies[triangles_inside_frustrum[:,0], :]
        #cube_triangle_norms = cube_triangle_norms.transpose()[triangles_inside_frustrum[:,0], :]
        if triangle_indicies.shape[0] == 0: 
            Screen.update(buffer_screen[:,:,:3]) #  If there are no triangles on screen then paint background or else last triangles stay on screen
            continue
        
        #   Perpective
        cube_verticies = Camera.local_to_persceptive_to_pixel_cordinates @ cube_verticies #   Note here we combine operations. Two transforms into one step = faster 
        #   Homogenous to normal vectors 
        cube_verticies[:2,:] /= cube_verticies[3,:] 
        #   Restore 4th dimension to place holder 1
        cube_verticies[3,:] = 1

        #   Recalculate norms post perspective transform
        cube_triangle_norms = Polygon.get_normals(cube_verticies[:3], triangle_indicies)
        
        #   Rasterize
        #   WARNING The normals are not recalculated with perspective. The perspective transform moves the vertices and angel of the faces
        cube_front_face_triangles = Rasterize.backface_culling(cube_triangle_norms.transpose())
        if cube_front_face_triangles.shape[0] == 0: 
            Screen.update(buffer_screen[:,:,:3]) #  If there are no triangles on screen then paint background or else last triangles stay on screen
            continue
        #   Cull back facing triangles 
        triangle_showing_indicies, triangle_showing_norms = triangle_indicies[cube_front_face_triangles], cube_triangle_norms[cube_front_face_triangles] 
        
        cube_verticies = Transforms.origin_center_to_topleft() @ cube_verticies # TODO as lineral alge Translate: local space to pixel space so origin is top left  
        triangles = np.array([cube_verticies[:, triangle] for triangle in triangle_showing_indicies])
        
        #   Texture Triangular mesh  TODO This is paralizable. Run each triangle on anther thread not core. 
        #for i, triangle in enumerate(triangle_showing_indicies):
        for i, triangle in enumerate(triangles):
    
            #triangle = cube_verticies[:, triangle]
            try:
                baycentric_transform = Transforms.to_barycentric_cartesian(triangle[[0,1,3]]) 
                #baycentric_transform = Transforms.to_barycentric_cartesian(triangle) 
            except: 
                """
                    If the triangle is orthagonal. Then two of its verticies will be scales of each other. Meaning that a primitive operation can remove a row
                    This means three variagles with two unique equations. Meaning, the solution is a line. Or parametrically, two of the variables are dependet on the third
                    Thus there is no inverse of baycentric_to_cartisian
                    Instead of drawing a line. I skip because of development time 
                    Infact back face culling will remove this outcome because the triangles normal is ortho to the cameras orientation
                """
                continue
            rectangle_max = np.ceil(np.max(triangle, axis=(1))).astype(int) #   We have zero error in finding which pixels are on when we go from decimal to int because we over engineer the rectange by 2 rows and 2 columns
            rectangle_min = np.floor(np.min(triangle, axis=(1))).astype(int)

            #   Storing the Screen identity in reshaped form is posible but the operations to index it make the benift not appealing
            rectangle_containing_triangle = Screen.SCREEN_IDENTITY[rectangle_min[0]:rectangle_max[0], rectangle_min[1]:rectangle_max[1]]

            #   Unknown reason why rectangles ever can have zero width or height when on edge of screen
            if rectangle_containing_triangle.shape[0] == 0 or rectangle_containing_triangle.shape[1] == 0: 
                continue

            #   We reshape the rect cordinates so that we can matrix multipy them into bayscentic cordinates
            reshape = (rectangle_containing_triangle.shape[0]*rectangle_containing_triangle.shape[1], 3) # NOTE the third dimension is 1. This is for baycentric transform, not z depth!
            reshaped_rectangle_containing_triangle = rectangle_containing_triangle.reshape(*reshape).transpose() 
            baycentric_rectangle = baycentric_transform @ reshaped_rectangle_containing_triangle  
            is_inside_triangle = (baycentric_rectangle >= 0).all(axis=0) #  Now we have a boolean matrix saying when each pixel is inside triangle 
            all_triangle_pixel_cordinates = reshaped_rectangle_containing_triangle[:, is_inside_triangle] # The cordinates of each pixel in triangle
            #   Get the texture scaled to the rectangle containing the triangle
            texture = Textures.reshape_texture_to_rectange(Textures.textures[object.uv_map[i]], rectange=rectangle_containing_triangle.shape[0:2]) 
            
            #   Get function to find z value of each pixel inside the triangle using equation of plane with z as only unknown and x and y as input
            find_z_in_triangle = Polygon.get_function_of_plane(triangle[:3,0], triangle_showing_norms[i])
            #   Set z value of each pixel
            all_triangle_pixel_cordinates[2,:] = [find_z_in_triangle(*pixel) for pixel in all_triangle_pixel_cordinates[:2,:].transpose()]
            
            #   Rasterize. We simply check if the buffered pixel is further away on z. If so swap with new value
            all_inside_triangle_and_not_overlaped_yet = \
                all_triangle_pixel_cordinates[2,:] <= buffer_screen[all_triangle_pixel_cordinates[0], all_triangle_pixel_cordinates[1], 3]
            all_showing_triangle_pixel_cordinates = all_triangle_pixel_cordinates[:, all_inside_triangle_and_not_overlaped_yet]
            
            #   Set the buffer pixels to the pixels of the texture
            buffer_screen[all_showing_triangle_pixel_cordinates[0], all_showing_triangle_pixel_cordinates[1], :3] = texture.reshape(*reshape)[is_inside_triangle][all_inside_triangle_and_not_overlaped_yet]
            #   Set pixels z depth so future pixels can overlap them if they are closer to the camera 
            buffer_screen[all_showing_triangle_pixel_cordinates[0], all_showing_triangle_pixel_cordinates[1], 3] = all_showing_triangle_pixel_cordinates[2]  

    Screen.update(buffer_screen[:,:,:3])

pygame.quit()
exit()









































        # square surfaces object_surface_indicies = [[0,1,2,3], [4,5,6,7], [0,1,4,5], [2,3,6,7], [0,3,4,7], [1,2,5,6]]
        #if object.cube_count != 1: raise Exception("This is a sigulton")
# t = Transforms()

# i, j, k = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
# ijk = np.array([i,j,k])
# #pi = np.pi 
# angles = [pi/2, 0, pi/2]
# print( np.round(t.euler_to_global_basis(*angles) @ i , 2))













# # Upgrade to quanternions for roations matrix
# def get_all_world_verticies():
#     pass

# def get_all_world_surfaces_sorted():
#     """ 
#         Sort each surface by its verticies lowest z value.
#         Surfaces are planes. Intersection of rigid body planes is a logical error sourced to the physics engine.
#         Other overlaps should have transluecent alphas. Meaning overlap is desired 
#     """
#     pass

# i, j, k = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
# ijk = np.array([i,j,k])
# object_v = np.array([\
#     [0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,1], [0,1,1], [1,1,1], [1,0,1] \
# ])

