SELECT * FROM comp_cast_type AS cct1, complete_cast AS cc, cast_info AS ci, char_name AS chn WHERE cct1.kind = 'cast' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;