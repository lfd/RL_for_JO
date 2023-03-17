SELECT * FROM title AS t, movie_info AS mi, cast_info AS ci, role_type AS rt, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND rt.role = 'actress' AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;