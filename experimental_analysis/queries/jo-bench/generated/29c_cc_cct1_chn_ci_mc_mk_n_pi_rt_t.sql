SELECT * FROM title AS t, char_name AS chn, name AS n, complete_cast AS cc, cast_info AS ci, movie_keyword AS mk, role_type AS rt, person_info AS pi, comp_cast_type AS cct1, movie_companies AS mc WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;