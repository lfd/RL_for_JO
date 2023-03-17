SELECT * FROM char_name AS chn, name AS n, complete_cast AS cc, cast_info AS ci, person_info AS pi, movie_keyword AS mk, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;