SELECT * FROM comp_cast_type AS cct1, name AS n, complete_cast AS cc, cast_info AS ci, person_info AS pi, movie_keyword AS mk, movie_info AS mi WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND n.gender = 'f' AND n.name LIKE '%An%' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;