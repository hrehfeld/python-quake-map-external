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

from subprocess import check_call, check_output


#from . import mathhelper

__version__ = '0.0.1'

basepath = Path('/home/hrehfeld/.quakespasm')
id1_path = basepath / 'id1'
compiler_base = Path('/home/hrehfeld/projects/map_compile/ericw-tools-v0.18.1-Linux/bin')

prefab_info_suffix = '.prefabinfo'


EPS = 0.00001

assumed_texture_size = 64


worldspawn_name = 'worldspawn'
map_external_name = 'misc_external_map'
map_external_angles_key = '_external_map_angles'
map_external_angles_yaw_key = '_external_map_angle'
map_external_filename_key = '_external_map'
wad_key = 'wad'
wad_sep = ';'
origin_key = 'origin'
origin_sep = ' '
angles_key = 'angles'
angles_yaw_key = 'angle'
angles_sep = ' '
angles_yaw_axis = 2
angles_roll_axis = 0
angles_pitch_axis = 1

prefab_name_prefix = 'prefab_'
prefab_name_key = 'prefab_name'

tb_def_key = '_tb_def'

brushes_key = 'brushes'

fgd_marker = '///prefabs -- do not edit after this line!\n'
fgd_prefab_marker = '///prefab: '

def warning(s):
    print('WARNING:', s)


def fgd_compile(path):
    cwd = id1_path
    out_path = path.with_suffix('.bsp')
    cmd = [compiler_base / 'qbsp', '-nopercent', '-nofill', '-noverbose', path, out_path]

    check_call(cmd, cwd=str(cwd))
    cmd = [compiler_base / 'light', '-light', '255'] + [out_path]
    check_call(cmd, cwd=str(cwd))
    return out_path


def bsp_get_aabb(bsp_path):
    cmd = [compiler_base / 'bsputil', '--check', str(bsp_path)]
    output = check_output(cmd, encoding='utf-8')
    output = output.split('\n')
    prefix = 'world mins: '
    # world mins: 1.000000 1.000000 1.000000 maxs: 31.000000 31.000000 63.000000
    for l in output:
        if l.startswith(prefix):
            l = l[len(prefix):]
            l = l.split()
            l.remove('maxs:')
            assert(len(l) == 6)

            return [float(s) for s in l]
    

def path_stem_add(p, s):
    return p.parent / (p.stem + s)
            

def split_value(v, sep=' ', type=str):
    return [type(s) for s in v.split(sep)]
    

def get_entity_key(ent, key, default=None):
    if not hasattr(ent, key):
        return default
    return getattr(ent, key)
    

def set_entity_key(ent, key, v):
    setattr(ent, key, v)


def set_entity_keys(ent, ks):
    for k, v in ks.items():
        set_entity_key(ent, k, v)


def del_entity_key(ent, key):
    if hasattr(ent, key):
        delattr(ent, key)


def get_entity_keys(e):
    r = odict((p for p in e.__dict__.items() if p[0] != brushes_key))
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


def set_entity_angles(ent, angles, key=angles_key, yaw_key=angles_yaw_key):
    def clean_angle(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return round(math.degrees(a))

    if abs(angles[angles_pitch_axis]) < EPS and abs(angles[angles_roll_axis]) < EPS:
        angle = (angles[angles_yaw_axis])
        if abs(angle) > EPS:
            set_entity_key(ent, yaw_key, clean_angle(angle))
        else:
            del_entity_key(ent, yaw_key)
        del_entity_key(ent, key)
    else:
        angles = angles[2 - angles_yaw_axis], angles[2 - angles_pitch_axis], angles[2 - angles_roll_axis]
        set_entity_vector(ent, key, [clean_angle(a) for a in angles], sep=angles_sep)
        del_entity_key(ent, yaw_key)
    

def origin_mat(e):
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

    #print(ent.classname, angles, 'degrees')

    return angles


def angles_to_mat(angles):
    angles = [-radians(float(a)) for a in angles]

    rotation = rotation_z_matrix(angles[0]) @ rotation_y_matrix(angles[1]) @ rotation_x_matrix(angles[2])

    return rotation
    
    

def clean_ent(e):
    for k, v in get_entity_keys(e).items():
        #print(k, v)
        if v == '""':
            delattr(e, k)

def get_wads(e):
    if not hasattr(e, wad_key):
        return []
    return getattr(e, wad_key).split(wad_sep)


def set_wads(e, wads):
    r = []
    for w in wads:
        if w not in r:
            r.append(w)

    if not r:
        return

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


def main():
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
        if t == worldspawn_name and not worldspawn:
            worldspawn = ent

        clean_ent(ent)

    fgd_prefabs = odict()

    # load fgd
    fgd_path = get_entity_key(worldspawn, tb_def_key)
    if fgd_path is None:
        warning('no custom fgd, can\'t add prefab_ defs')
        fgd = None
    else:
        prefix = 'external:'
        assert(fgd_path.startswith(prefix))
        fgd_path = fgd_path[len(prefix):]

        with Path(fgd_path).open('r') as f:
            fgd = f.read()

        end = fgd.rfind(fgd_marker)
        defs = fgd[end:]

        for l in defs.split('\n'):
            if l.startswith(fgd_prefab_marker):
                l = l.split(' ')
                marker, name, path = l
                assert(name not in fgd_prefabs)
                fgd_prefabs[name] = id1_path / Path(path)


        if end is not None:
            fgd = fgd[:end]
        

    assert(worldspawn)

    temp_sub_prefabs = odict()
    fgd_models = odict()
    fgd_maps = odict()

    # make a copy of list so we can append to base_map
    for ient, ent in ((i, e) for i, e in enumerate(list(base_map))):
        t = ent.classname
        print('===============next entity', ient, t)

        if ent.classname.startswith(prefab_name_prefix):
            name = ent.classname[len(prefab_name_prefix):]
            path = fgd_prefabs[name]
            print('converting %s to %s with %s' % (ent.classname, map_external_name, path), name)
            set_entity_key(ent, map_external_filename_key, path)
            ent.classname = map_external_name

        if ent.classname != map_external_name:
            continue

        ks = get_entity_keys(ent)
        if map_external_filename_key in ks:
            base_map.remove(ent)
            ext_map_path = base_filepath.parent / (ks[map_external_filename_key])
            ent_angles = parse_angles(ent)
            ent_rotation_mat = angles_to_mat(ent_angles)
            ent_origin = get_entity_origin(ent)
            ent_mat = origin_mat(ent) @ ent_rotation_mat

            if not ext_map_path.exists():
                warning('external map %s doesn\'t exist!' % ext_map_path)
                continue
            # everything prefixed with o is from the external map
            with ext_map_path.open('r') as f:
                s = f.read()
            omap = m.loads(s)

            oworldspawn = omap[0]

            prefab_name = get_entity_key(oworldspawn, prefab_name_key, ext_map_path.stem)
            if ' ' in prefab_name:
                warning('%s: prefab_name (%s) cannot contain space (" ")' % (ext_map_path, prefab_name))
                prefab_name = ext_map_path.stem

            fgd_model_worldspawn = m.Entity()
            fgd_model_worldspawn.classname = worldspawn_name

            wads = []
            for ioent, oent in enumerate(omap):
                oent_origin = get_entity_origin(oent)
                oent_angles = parse_angles(oent)
                oent_mat = angles_to_mat(oent_angles) @ mathhelper.Matrices.translation_matrix(*oent_origin)
                oent_mat = oent_mat @ ent_mat

                wads += get_wads(oent)

                _angles_key = angles_key
                _angles_yaw_key = angles_yaw_key
                if oent.brushes:
                    fgd_model_worldspawn.brushes += oent.brushes

                    #bs = [transform_brush(oent_mat, b) for b in bs]
                    # ot = oent.classname
                    # if ot == worldspawn_name:
                    #     # TODO: copy ent keys?
                    #     worldspawn.brushes += oent.brushes

                    oent_base_path = path_stem_add(ext_map_path, '_' + oent.classname)

                    # don't overwrite if e.g. two func_door in the prefab
                    oent_path = oent_base_path
                    i = 0
                    while oent_path in temp_sub_prefabs and temp_sub_prefabs[oent_path] != ext_map_path:
                        oent_path = oent_base_path + str(ioent)
                        i += 1
                    out_p = oent_path.with_suffix('.temp.map')
                    out_p_s = str(out_p)

                    if out_p_s in temp_sub_prefabs:
                        assert(temp_sub_prefabs[out_p_s] == ext_map_path)
                    else:
                        e = m.Entity()
                        e.classname = worldspawn_name
                        e.brushes = oent.brushes

                        s = m.dumps([e])

                        print('writing %s' % out_p)
                        with out_p.open('w') as f:
                            f.write(s)

                        temp_sub_prefabs[out_p_s] = ext_map_path

                    e = m.Entity()
                    oks = get_entity_keys(oent)
                    set_entity_keys(e, oks)

                    e.classname = map_external_name
                    e._external_map = str(out_p)
                    e._external_map_classname = oent.classname
                    if 'angles' in oks:
                        e._external_map_angles = oent.angles
                    #e._external_map_scale =

                    _angles_key = map_external_angles_key
                    _angles_yaw_key = map_external_angles_yaw_key


                    oent = e

                set_entity_origin(oent, transform_point(ent_mat, oent_origin))
                set_entity_angles(oent, extract_angles(oent_mat), key=_angles_key, yaw_key=_angles_yaw_key)

                if oent.classname != worldspawn_name:
                    base_map.append(oent)

            # copy wad info from all ents
            set_wads(worldspawn, get_wads(worldspawn) + wads)

            # dump into fgd later
            # make sure duplicate names all reference the same prefab
            if prefab_name in fgd_maps:
                print(fgd_maps[prefab_name], ext_map_path)
                assert(fgd_maps[prefab_name] == ext_map_path)
            if ' ' in prefab_name:
                raise Exception('%s: prefab_name (%s) cannot contain space (" ")' % (ext_map_path, prefab_name))
            
            # compile external map for fgd
            if prefab_name not in fgd_models:
                set_wads(fgd_model_worldspawn, wads)
                fgd_models[prefab_name] = fgd_model_worldspawn
                print(fgd_model_worldspawn.wad)
                
                
            fgd_maps[prefab_name] = (ext_map_path)

    if fgd is not None:
        defs = []
        for name, (map_path) in fgd_maps.items():
            model_worldspawn = fgd_models[name]
            #print(name, model_worldspawn)
            assert(map_path.exists())

            fgd_model = m.dumps([model_worldspawn])
            fgd_model_path = path_stem_add(map_path, '-fgd').with_suffix('.map')
            with fgd_model_path.open('w') as f:
                f.write(fgd_model)
            model_path = fgd_compile(fgd_model_path)
            assert(model_path.exists())

            
            aabb = bsp_get_aabb(model_path)
            vs = [abs(v) for v in aabb[:3]]
            for i in range(3):
                x = abs(aabb[3 + i])
                vs[i] += x
                vs[i] *= 0.5
            aabb = [-v for v in vs] + vs
            aabb = [str(s) for s in aabb]
            aabb = ' '.join(aabb[:3]) + ', ' + ' '.join(aabb[3:])
            #print(aabb)
            
            map_path = map_path.resolve().relative_to(id1_path.resolve())
            model_path = model_path.resolve().relative_to(id1_path.resolve())

            vs = ['map_path(string) : "path to the source map file" : "%s" : "Long description"' % map_path]
            vs += ['%s(string) : "%s (Pitch Yaw Roll)"' % (angles_key, angles_key)]
            vs = '\n'.join(vs)


            d = '@PointClass size(%s) color(200 200 0) studio("%s") = %s : "%s" [%s]'  % (aabb, model_path, prefab_name_prefix + name, model_path, vs)
            c = fgd_prefab_marker + '%s %s' % (name, map_path)
            defs.append(d)
            defs.append(c)

        defs += [fgd_prefab_marker + name + ' ' + str(path.resolve().relative_to(id1_path.resolve())) for name, path in fgd_prefabs.items() if name not in fgd_maps]

        defs = '\n'.join(defs)

        fgd += fgd_marker + defs
        with Path(fgd_path).open('w') as f:
            f.write(fgd)




    #print(omap)
    #print(getattr(worldspawn, wad_key))
    s = m.dumps(base_map)

    with out_filepath.open('w') as f:
        f.write(s)

main()
