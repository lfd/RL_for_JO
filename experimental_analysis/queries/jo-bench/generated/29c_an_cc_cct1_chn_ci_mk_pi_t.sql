SELECT * FROM movie_keyword AS mk, person_info AS pi, title AS t, char_name AS chn, complete_cast AS cc, cast_info AS ci, comp_cast_type AS cct1, aka_name AS an WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;