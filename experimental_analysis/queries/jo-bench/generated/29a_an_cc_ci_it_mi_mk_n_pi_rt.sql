SELECT * FROM name AS n, person_info AS pi, movie_keyword AS mk, cast_info AS ci, role_type AS rt, aka_name AS an, movie_info AS mi, complete_cast AS cc, info_type AS it WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id;