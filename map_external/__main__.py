import argparse
import json
import os
import sys
import time

import random

import numpy

# disable 'scientific' notation
numpy.set_printoptions(suppress=True)
#import tmx

from quake import map as m

from pathlib import Path

from math import radians
import math

from collections import OrderedDict as odict

import mathhelper

#from . import mathhelper

__version__ = '0.0.1'

EPS = 0.00001

assumed_texture_size = 64


worldspawn_name = 'worldspawn'
map_external_name = 'misc_external_map'
external_map_filename_key = '_external_map'
wad_key = 'wad'
wad_sep = ';'
origin_key = 'origin'
origin_sep = ' '
angles_key = 'mangle'
angles_yaw_key = 'angle'
angles_sep = ' '
angles_yaw_axis = 0
angles_roll_axis = 2
angles_pitch_axis = 1

brushes_key = 'brushes'

def warning(s):
    print('WARNING:', s)


def split_value(v, sep=' ', type=str):
    return [type(s) for s in v.split(sep)]
    

def get_entity_key(ent, key, default=None):
    if not hasattr(ent, key):
        return default
    return getattr(ent, key)
    

def set_entity_key(ent, key, v):
    setattr(ent, key, v)


def del_entity_key(ent, key):
    if hasattr(ent, key):
        delattr(ent, key)


def get_entity_keys(e):
    print(e.__dict__)
    r = odict((p for p in e.__dict__.items() if p[0] != brushes_key))
    print(r)
    return r


def get_entity_origin(e):
    o = get_entity_key(e, origin_key)
    if o is None:
        return 0, 0, 0
    r = split_value(o, sep=origin_sep, type=float)
    assert(len(r) == 3)
    return r


def set_entity_vector(ent, key, v, sep=origin_sep):
    setattr(ent, key, sep.join([str(round(x)) for x in v]))


def set_entity_origin(ent, origin):
    origin = [round(o) for o in origin]
    set_entity_vector(ent, origin_key, origin, sep=origin_sep)


def set_entity_angles(ent, mat):
    angles = extract_angles(mat)

    def clean_angle(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return round(math.degrees(a))

    if abs(angles[angles_pitch_axis]) < EPS and abs(angles[angles_roll_axis]) < EPS:
        angle = (angles[angles_yaw_axis])
        if abs(angle) > EPS:
            set_entity_key(ent, angles_yaw_key, clean_angle(angle))
        else:
            del_entity_key(ent, angles_yaw_key)
        del_entity_key(ent, angles_key)
    else:
        angles = angles[2 - angles_yaw_axis], angles[2 - angles_pitch_axis], angles[2 - angles_roll_axis]
        set_entity_vector(ent, angles_key, [clean_angle(a) for a in angles], sep=angles_sep)
        del_entity_key(ent, angles_yaw_key)
    

def parse_origin(e):
    origin = get_entity_origin(e)

    return mathhelper.Matrices.translation_matrix(*origin)
    

def parse_angles(ent):
    _angles = 0, 0, 0
    if hasattr(ent, angles_key):
        a = getattr(ent, angles_key)
        _angles = a.split(angles_sep)
    elif hasattr(ent, angles_yaw_key):
        a = getattr(ent, angles_yaw_key)
        _angles = a, 0, 0
    if len(_angles) != 3:
        raise Exception('malformed angles: %s' % _angles)
        
    else:        
        angles = _angles

    print(ent.classname, angles, 'degrees')
        
    angles = [-radians(float(a)) for a in angles]

    rotation = rotation_z_matrix(angles[0]) @ rotation_y_matrix(angles[1]) @ rotation_x_matrix(angles[2])

    return rotation
    
    

def clean_ent(e):
    for k, v in get_entity_keys(e).items():
        print(k, v)
        if v == '""':
            delattr(e, k)

def get_wads(e):
    if not hasattr(e, wad_key):
        return []
    return getattr(e, wad_key).split(wad_sep)


def set_wads(e, wads):
    if not wads:
        return

    r = []
    for w in wads:
        if w not in r:
            r.append(w)

    return setattr(e, wad_key, wad_sep.join(r))


def get_dominant_axis(points):
    p0, p1, p2 = points[:3]
    plane_normal = get_plane_normal(points)

    # Determine vector component with largest magnitude
    dominant_axis = 0
    for i, x in enumerate(plane_normal):
        if abs(x) >= abs(plane_normal[dominant_axis]):
            dominant_axis = i
    
    return dominant_axis, plane_normal[dominant_axis]


def get_axis_proj_indices(dominant_axis):
    return (
        (1, 2, 0)
        , (0, 2, 1)
        , (0, 1, 2)
    )[dominant_axis]


def get_axis_swizzle_mat(dominant_axis, id_mat=None):
    if id_mat is None:
        id_mat = numpy.identity(4)
    
    ins = get_axis_proj_indices(dominant_axis)
    r = numpy.array((
        id_mat[ins[0]]
        , id_mat[ins[1]]
        , id_mat[ins[2]]
        , id_mat[3]
    ))
    return r


def get_axis_project_mat(dominant_axis, magnitude=1.0):
    id_mat = numpy.identity(4)

    ins = get_axis_proj_indices(dominant_axis)

    muls = [1, 1, 1, 1]

    # up is always flipped
    #muls[ins[2]] = -1
    
    # flip left axis
    if magnitude < 0:
        muls[ins[0]] = -1

        print('flipping', dominant_axis, magnitude, '=> flip axis', ins[0])
    cols = [muls[i] * id_mat[i] for i in range(4)]
    id_mat = numpy.array((cols[0], cols[1], cols[2], cols[3]))

    r = get_axis_swizzle_mat(dominant_axis, id_mat=id_mat)
    return r


def get_world_to_tex_space(plane):
    axis_proj = get_axis_project_mat(*get_dominant_axis(plane.points))
    #axis_proj = get_axis_swizzle_mat(get_dominant_axis(plane.points)[0])

    translation = mathhelper.Matrices.translation_matrix(*plane.offset, 0)
    scale = mathhelper.Matrices.scale_matrix(*(1.0 / s for s in plane.scale))
    rotation = rotation_z_matrix(math.radians(plane.rotation))

    return translation @ rotation @ scale @ axis_proj


def rotation_x_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array((
        (1.0, 0.0, 0.0, 0.0),
        (0.0,   c,   s, 0.0),
        (0.0,  -s,   c, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))


def rotation_y_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array((
        (c,   0.0,  -s, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (s,  0.0,    c, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))


def rotation_z_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array((
        (  c,   s, 0.0, 0.0),
        ( -s,   c, 0.0, 0.0),
        (0.0,  0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))



def transform_point(mat, p):
    return tuple(numpy.dot(mat, (*p, 1))[:3])



def reconstruct_proj_tex_mat(projected_points, desired_uvs):
    assert(len(projected_points) == len(desired_uvs) == 3)

    #  u = x * m00 + y * m01 + z * m02 + w * m03
    #  v = x * m10 + y * m11 + z * m12 + w * m13
    # _u = x * m20 + y * m21 + z * m22 + w * m23
    # _v = x * m30 + y * m31 + z * m32 + w * m33

    # u0 = x0 * m00 + y0 * m01 + w0 * m02
    # u1 = x1 * m00 + y1 * m01 + w1 * m02
    # u2 = x2 * m00 + y2 * m01 + w2 * m02

    # u0 = x0 * m00 + y0 * m01 + z0 * m02 + m03
    # u1 = x1 * m00 + y1 * m01 + z1 * m02 + m03
    # u2 = x2 * m00 + y2 * m01 + z2 * m02 + m03
    # u3 = x3 * m00 + y3 * m01 + z3 * m02 + m03

    # w is always 1, so:
    def get_uv_mat_row(uvs, *points):
        u0, u1, u2 = uvs
        (x0, y0), (x1, y1), (x2, y2) = points
        m00 = -(u0 * y1 - u0 * y2 - u1 * y0 + u1 * y2 + u2 * y0 - u2 * y1) / (-x0 * y1 + x0 * y2 + x1 * y0 - x1 * y2 - x2 * y0 + x2 * y1)
        m01 = -(-u0 * x1 + u0 * x2 + u1 * x0 - u1 * x2 - u2 * x0 + u2 * x1) / (-x0 * y1 + x0 * y2 + x1 * y0 - x1 * y2 - x2 * y0 + x2 * y1)
        m02 = -(u0 * x1 * y2 - u0 * x2 * y1 - u1 * x0 * y2 + u1 * x2 * y0 + u2 * x0 * y1 - u2 * x1 * y0) / (-x0 * y1 + x0 * y2 + x1 * y0 - x1 * y2 - x2 * y0 + x2 * y1)
        return m00, m01, 0, m02

    #def get_uv_mat_row(us, p0, p1, p2):
        # P = numpy.array((
        #     p0, p1, p2
        # ))

        # print(P)
        # P_inv = numpy.linalg.inv(P)
        # print(P_inv)

        # m = P_inv @ us



    us = [uv[0] for uv in desired_uvs]
    vs = [uv[1] for uv in desired_uvs]

    world_tex_mat_2d = numpy.array((
        # solve for u
        get_uv_mat_row(us, *projected_points)
        # solve for v
        , get_uv_mat_row(vs, *projected_points)
        , (0.0, 0.0, 0.0, 0.0)
        , (0.0, 0.0, 0.0, 1.0)
    ))

    return world_tex_mat_2d


def get_plane_normal(points):
    p0, p1, p2 = points
    plane_normal = numpy.cross(
        numpy.subtract(p2, p0),
        numpy.subtract(p1, p0)
    )
    plane_normal = plane_normal / numpy.linalg.norm(plane_normal)
    return plane_normal


def get_plane_normal_form(points):
    n = get_plane_normal(points)

    p = points[0]
    d = -n[0] * p[0] - n[1] * p[1] - n[2] * p[2]
    return n, d


def get_intersection_point(planes):
    ((a0, b0, c0), d0), ((a1, b1, c1), d1), ((a2, b2, c2), d2) = planes

    x = -(-b0 * c1 * d2 + b0 * c2 * d1 + b1 * c0 * d2 - b1 * c2 * d0 - b2 * c0 * d1 + b2 * c1 * d0) / (-a0 * b1 * c2 + a0 * b2 * c1 + a1 * b0 * c2 - a1 * b2 * c0 - a2 * b0 * c1 + a2 * b1 * c0)
    y = -(a0 * c1 * d2 - a0 * c2 * d1 - a1 * c0 * d2 + a1 * c2 * d0 + a2 * c0 * d1 - a2 * c1 * d0) / (-a0 * b1 * c2 + a0 * b2 * c1 + a1 * b0 * c2 - a1 * b2 * c0 - a2 * b0 * c1 + a2 * b1 * c0)
    z = -(-a0 * b1 * d2 + a0 * b2 * d1 + a1 * b0 * d2 - a1 * b2 * d0 - a2 * b0 * d1 + a2 * b1 * d0) / (-a0 * b1 * c2 + a0 * b2 * c1 + a1 * b0 * c2 - a1 * b2 * c0 - a2 * b0 * c1 + a2 * b1 * c0)
    return x, y, z


def get_obj(brush, uvs):
    planes = [get_plane_normal_form(pl.points) for pl in brush.planes]
    


def transform_brush(local_world_mat, b):
    print('BRUUUUUUUUUUUUUUUUUUUUUUUUSH')
    for plane in b.planes:
        assert(len(plane.points) >= 3)

        for s in plane.scale:
            assert(s > 0)

        print('--------------')

        old_dominants = get_dominant_axis(plane.points)

        print('points', plane.points)
        print('face', plane.offset, plane.rotation, plane.scale, plane.texture_name)
        world_tex_mat = get_world_to_tex_space(plane)
        print(world_tex_mat)

        old_uvs = [transform_point(world_tex_mat, p) for p in plane.points]
        print('old uvs', old_uvs)

        #get_obj(b, old_uvs)

        plane.points = [transform_point(local_world_mat, p) for p in plane.points]
        print('transformed points', plane.points)

        dominant_axis, dominant_magnitude = get_dominant_axis(plane.points[:3])
        axis_proj = get_axis_project_mat(dominant_axis, dominant_magnitude)
        axis_swizzle = get_axis_swizzle_mat(dominant_axis)
        axis_swizzle_inv = numpy.linalg.inv(axis_swizzle)

        # only use the 2 dimensions of the axis projection for solving
        proj_points = [transform_point(axis_proj, p)[:2] for p in plane.points]
        print(proj_points)

        proj_tex_mat = reconstruct_proj_tex_mat(proj_points, old_uvs)

        world_tex_mat = proj_tex_mat @ axis_proj

        zero = transform_point(proj_tex_mat, (0, 0, 0))[:2]
        one = transform_point(proj_tex_mat, (1, 1, 0))[:2]

        plane.offset = zero[0], zero[1]
        angles = extract_angles(proj_tex_mat)
        plane.rotation = -math.degrees(angles[2])

        left = numpy.subtract(transform_point(proj_tex_mat, (1, 0, 0))[:2], zero)
        up = numpy.subtract(transform_point(proj_tex_mat, (0, 1, 0))[:2], zero)
        print('plane vecs', zero, one, left, up)
        plane.scale = [1.0 / a for a in (numpy.linalg.norm(left), numpy.linalg.norm(up))]


        print('solved world->tex')
        print(world_tex_mat)
        print(proj_tex_mat)
        print(proj_tex_mat)
        print(axis_proj)
        print(axis_swizzle)
        print(axis_swizzle_inv)
        
        print('face', plane.offset, plane.rotation, plane.scale)

        for i, p in enumerate(plane.points):
            uv = transform_point(world_tex_mat, p)
            d = numpy.subtract(old_uvs[i], uv)
            l = numpy.linalg.norm(d[:2])
            if l >= EPS:
                raise Exception(i, p, d, uv, old_uvs[i])

        test_mat = get_world_to_tex_space(plane)
        print('reconstructed mat')
        print(test_mat)
        for i, p in enumerate(plane.points):
            uv = transform_point(test_mat, p)
            d = numpy.subtract(old_uvs[i], uv)
            l = numpy.linalg.norm(d[:2])
            if l >= EPS:
                raise Exception(d, uv, old_uvs[i])


        

    return b


def extract_angles(mat):
    atan2 = math.atan2

    def m(i, j):
        return mat.item((j, i))

    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
    t1 = atan2(m(1, 2), m(2, 2))
    c2 = math.sqrt(m(0, 0) ** 2 + m(0, 1) ** 2)
    t2 = atan2(-m(0, 2), c2)
    s1 = math.sin(t1)
    c1 = math.cos(t1)
    t3 = atan2(s1 * m(2, 0) - c1 * m(1, 0), c1 * m(1, 1) - s1 * m(2, 1))
    return t1, t2, t3

parser = argparse.ArgumentParser(prog='map_external',
                description='TODO',
                epilog='TODO')

parser.add_argument('map', help='base map')
parser.add_argument('out', help='output map')

parser.add_argument('-v',
                    '--version',
                    dest='version',
                    action='version',
                    version='%(prog)s {}'.format(__version__),
                    help='display version number')

parser.add_argument('-q',
                    '--quiet',
                    dest='quiet',
                    action='store_true',
                    help='quiet mode')

args = parser.parse_args()


base_filepath = Path(args.map)
out_filepath = Path(args.out)

with base_filepath.open('r') as f:
    base_map = m.loads(f.read())


worldspawn = None

for ent in base_map:
    t = ent.classname
    if t == worldspawn_name:
        worldspawn = ent

    clean_ent(ent)
    print(t)

assert(worldspawn)
print(worldspawn)


    

for ient, ent in ((i, e) for i, e in enumerate(base_map) if e.classname == map_external_name):
    t = ent.classname
    print('===============next entity', ient, t)



    ks = get_entity_keys(ent)
    if external_map_filename_key in ks:
        p = base_filepath.parent / (ks[external_map_filename_key])
        if not p.exists():
            warning('external map %s doesn\'t exist!' % p)
            continue
        else:
            ent_rotation_mat = parse_angles(ent)
            ent_mat = parse_origin(ent) @ ent_rotation_mat

            # everything prefixed with o is from the external map
            with p.open('r') as f:
                omap = m.loads(f.read())

            for ioent, oent in enumerate(omap):
                # copy wad info from all ents
                set_wads(worldspawn, get_wads(worldspawn) + get_wads(oent))


                oent_origin = get_entity_origin(oent)
                set_entity_origin(oent, transform_point(ent_mat, oent_origin))
                print(oent.classname)
                oent_mat = parse_angles(oent) @ mathhelper.Matrices.translation_matrix(*oent_origin)
                print(oent_mat)
                print([math.degrees(a) for a in extract_angles(ent_mat)])
                print([math.degrees(a) for a in extract_angles(ent_rotation_mat)])
                print([math.degrees(a) for a in extract_angles(oent_mat)])
                print([math.degrees(a) for a in extract_angles(ent_mat @ oent_mat)])
                print(oent_origin)
                oent_mat =  oent_mat @ ent_mat

                set_entity_angles(oent, (oent_mat))

                bs = oent.brushes
                bs = [transform_brush(oent_mat, b) for b in bs]
                oent.brushes = bs

                ot = oent.classname
                if ot == worldspawn_name:
                    # TODO: copy ent keys?
                    worldspawn.brushes += oent.brushes
                else:
                    base_map.insert(ient, oent)

        base_map.remove(ent)


print(omap)
print(getattr(worldspawn, wad_key))
s = m.dumps(base_map)
    
with out_filepath.open('w') as f:
    f.write(s)
