SELECT * FROM title AS t, complete_cast AS cc, kind_type AS kt, movie_companies AS mc, company_name AS cn, movie_info AS mi WHERE cn.country_code = '[us]' AND kt.kind IN ('movie', 'tv movie', 'video movie', 'video game') AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 1990 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;