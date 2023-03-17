SELECT * FROM kind_type AS kt, title AS t, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND kt.kind = 'movie' AND t.title != '' AND (t.title LIKE '%Champion%' OR t.title LIKE '%Loser%') AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;