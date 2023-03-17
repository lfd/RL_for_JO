SELECT * FROM complete_cast AS cc, movie_info AS mi, comp_cast_type AS cct1 WHERE cct1.kind = 'complete+verified' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;